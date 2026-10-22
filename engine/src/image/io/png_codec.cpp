#include <csetjmp>
#include <cstring>
#include <memory>
#include <vector>

#include <png.h>

#include "base/utils/error.hpp"
#include "image/io/codec.hpp"

namespace alphashape::image_detail {

namespace {

/// libpng's error path is a longjmp: the error handler is expected to jump, so
/// returning from it makes libpng fall through to its own default handler,
/// which prints "libpng error: ..." to stderr and longjmps anyway. Jumping
/// here instead keeps a test run silent.
void png_silent_error(png_structp png, png_const_charp /*msg*/) {
    png_longjmp(png, 1);
}
void png_silent_warning(png_structp /*png*/, png_const_charp /*msg*/) {
    // Warnings are ignored; they do not indicate a failed decode.
}

/// Memory reader: hands libpng the bytes we already read off disk.
struct PngReader {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t offset = 0;
};

void png_mem_read(png_structp png, png_bytep out, png_size_t count) {
    PngReader* reader = static_cast<PngReader*>(png_get_io_ptr(png));
    if (count > reader->size - reader->offset) {
        png_error(png, "PNG: truncated data");
        return;
    }
    std::memcpy(out, reader->data + reader->offset, count);
    reader->offset += count;
}

/// Everything the decode touches lives behind this one heap pointer. It is
/// created before the setjmp and never reassigned, so a longjmp out of libpng
/// leaves no automatic object with an indeterminate value behind; the throw
/// then unwinds normally and the destructor frees the read structs.
struct PngState {
    png_structp png = nullptr;
    png_infop info = nullptr;
    PngReader reader;
    /// Channel count and bit depth as stored in the file, captured before the
    /// transform pass rewrites the layout.
    int stored_channels = 0;
    int stored_bit_depth = 0;
    std::vector<png_bytep> rows;
    /// Decoded rows. Must live here, not as a local in png_decode: a local
    /// constructed after setjmp would have its destructor skipped by libpng's
    /// longjmp, which is undefined behaviour.
    std::vector<png_byte> buffer;

    ~PngState() {
        if (png != nullptr) {
            png_destroy_read_struct(&png, info != nullptr ? &info : nullptr, nullptr);
        }
    }
};

/// Allocates the read/info structs. Separate from the header read because the
/// jmp_buf only exists once png_create_read_struct has succeeded; the caller
/// arms setjmp in between the two.
void create_structs(PngState& state) {
    state.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, png_silent_error,
                                       png_silent_warning);
    if (state.png == nullptr) {
        throw ProjectIOError("PNG: cannot allocate a decoder");
    }
    state.info = png_create_info_struct(state.png);
    if (state.info == nullptr) {
        throw ProjectIOError("PNG: cannot allocate image info");
    }
}

/// Parses the header and applies the transforms that make the pixel layout
/// predictable: palette to RGB, tRNS to full alpha, sub-8-bit grey widened to
/// 8-bit. Alpha is *not* stripped — the conversion loop drops it while it also
/// performs 16-bit rounding, in a single pass.
void read_header_and_transform(PngState& state, const std::uint8_t* data, std::size_t size) {
    state.reader.data = data;
    state.reader.size = size;
    state.reader.offset = 0;
    png_set_read_fn(state.png, &state.reader, png_mem_read);
    png_read_info(state.png, state.info);

    // "channels as stored": the file's own sample count. A 2-sample grey+alpha
    // file reports 1 (alpha is not a colour channel here), while a 4-sample
    // RGBA file reports 4, matching the interface's "1, 3, or 4 with alpha".
    const int raw = static_cast<int>(png_get_channels(state.png, state.info));
    state.stored_channels = raw == 2 ? 1 : raw;

    const png_byte colour_type = png_get_color_type(state.png, state.info);
    const png_byte bit_depth = png_get_bit_depth(state.png, state.info);
    state.stored_bit_depth = static_cast<int>(bit_depth);
    if (colour_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(state.png);
    }
    if (colour_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(state.png);
    }
    if (png_get_valid(state.png, state.info, PNG_INFO_tRNS) != 0) {
        png_set_tRNS_to_alpha(state.png);
    }
    png_read_update_info(state.png, state.info);
}

}  // namespace

ImageInfo png_info(const std::uint8_t* data, std::size_t size) {
    const std::unique_ptr<PngState> state(new PngState());
    create_structs(*state);
    if (setjmp(png_jmpbuf(state->png)) != 0) {
        throw ProjectIOError("PNG: malformed file");
    }
    read_header_and_transform(*state, data, size);

    ImageInfo info;
    info.format = ImageFormat::Png;
    info.width = static_cast<int>(png_get_image_width(state->png, state->info));
    info.height = static_cast<int>(png_get_image_height(state->png, state->info));
    info.channels = state->stored_channels;
    // Report the file's own bit depth, not the transformed one: a 2-bit grey
    // PNG is stored as 2 bits even though we widen it to 8 for the Image.
    info.bit_depth = state->stored_bit_depth;
    return info;
}

Image png_decode(const std::uint8_t* data, std::size_t size) {
    const std::unique_ptr<PngState> state(new PngState());
    create_structs(*state);
    if (setjmp(png_jmpbuf(state->png)) != 0) {
        throw ProjectIOError("PNG: malformed file");
    }
    read_header_and_transform(*state, data, size);

    const png_uint_32 width = png_get_image_width(state->png, state->info);
    const png_uint_32 height = png_get_image_height(state->png, state->info);
    const png_byte colour_type = png_get_color_type(state->png, state->info);
    const bool sixteen = png_get_bit_depth(state->png, state->info) == 16;

    // Reject an implausible geometry before buffers are sized from it. The PNG
    // limit is 2^31-1 per side; a corrupt IHDR can still claim far more than any
    // machine will hold, and libpng would try to allocate it for us.
    check_decodable(width, height, 8, "PNG");

    // After the transforms the colour type is one of these four; anything else
    // is a layout we do not decode.
    if (colour_type != PNG_COLOR_TYPE_GRAY && colour_type != PNG_COLOR_TYPE_GRAY_ALPHA &&
        colour_type != PNG_COLOR_TYPE_RGB && colour_type != PNG_COLOR_TYPE_RGB_ALPHA) {
        throw ProjectIOError("PNG: unsupported colour type");
    }
    const bool grey = colour_type == PNG_COLOR_TYPE_GRAY || colour_type == PNG_COLOR_TYPE_GRAY_ALPHA;
    const int out_components = grey ? 1 : 3;

    const int src_components = static_cast<int>(png_get_channels(state->png, state->info));
    const int sample_bytes = sixteen ? 2 : 1;
    const int row_bytes = static_cast<int>(png_get_rowbytes(state->png, state->info));
    if (src_components < out_components) {
        throw ProjectIOError("PNG: source has fewer channels than the destination");
    }

    // One contiguous buffer for the whole image, so the conversion loop indexes
    // it with a plain stride and no per-row allocation is needed.
    std::vector<png_byte>& buffer = state->buffer;
    buffer.resize(static_cast<std::size_t>(row_bytes) * static_cast<std::size_t>(height));
    if (!buffer.empty()) {
        state->rows.resize(height);
        for (png_uint_32 y = 0; y < height; ++y) {
            state->rows[y] =
                buffer.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_bytes);
        }
        png_read_image(state->png, state->rows.data());
    }
    png_read_end(state->png, nullptr);

    // libpng can no longer longjmp past this point.
    Image image(static_cast<int>(width), static_cast<int>(height), out_components);
    for (png_uint_32 y = 0; y < height; ++y) {
        const png_byte* row =
            buffer.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_bytes);
        for (png_uint_32 x = 0; x < width; ++x) {
            for (int c = 0; c < out_components; ++c) {
                const std::size_t offset =
                    (static_cast<std::size_t>(x) * static_cast<std::size_t>(src_components) +
                     static_cast<std::size_t>(c)) *
                    static_cast<std::size_t>(sample_bytes);
                std::uint32_t value = 0;
                if (sixteen) {
                    // libpng returns 16-bit samples big-endian unless
                    // png_set_swap is requested; compose explicitly so the code
                    // never depends on the host's byte order.
                    value = (static_cast<std::uint32_t>(row[offset]) << 8) |
                            static_cast<std::uint32_t>(row[offset + 1]);
                    value = (value * 255U + 32767U) / 65535U;  // round, not truncate
                } else {
                    value = row[offset];
                }
                image.at(static_cast<int>(x), static_cast<int>(y), c) =
                    static_cast<std::uint8_t>(value);
            }
        }
    }
    return image;
}

namespace {

/// Memory writer: libpng appends through this into the caller's byte vector.
/// The vector lives in the EncodeState below, not as a local constructed after
/// setjmp, so a longjmp out of libpng cannot skip a destructor.
void png_mem_write(png_structp png, png_bytep data, png_size_t count) {
    std::vector<std::uint8_t>* out =
        static_cast<std::vector<std::uint8_t>*>(png_get_io_ptr(png));
    out->insert(out->end(), data, data + count);
}

void png_mem_flush(png_structp /*png*/) {}

/// Everything the encode touches lives behind this one heap pointer, created
/// before the setjmp and never reassigned. The output vector is a member, so a
/// longjmp leaves no automatic object with an indeterminate value behind: the
/// throw then unwinds normally and the destructor frees the write structs.
struct EncodeState {
    png_structp png = nullptr;
    png_infop info = nullptr;
    std::vector<std::uint8_t> out;
    /// Row pointers handed to png_write_image. Must live here, not as a local
    /// constructed after setjmp: that local's destructor would be skipped by
    /// libpng's longjmp, which is undefined behaviour.
    std::vector<png_bytep> rows;

    ~EncodeState() {
        if (png != nullptr) {
            png_destroy_write_struct(&png, info != nullptr ? &info : nullptr);
        }
    }
};

}  // namespace

std::vector<std::uint8_t> png_encode(const Image& image) {
    if (image.empty()) {
        throw ValueError("PNG: cannot encode an empty image");
    }
    const int channels = image.channels();
    if (channels != 1 && channels != 3 && channels != 4) {
        throw ValueError("PNG: can only encode 1, 3 or 4 channel images");
    }
    // The PNG format itself caps a side at 2^31-1; staying under the decoder's
    // own kMaxDimension keeps every file we write one we can also read back.
    if (static_cast<std::uint64_t>(image.width()) > kMaxDimension ||
        static_cast<std::uint64_t>(image.height()) > kMaxDimension) {
        throw ValueError("PNG: image dimensions exceed the supported maximum");
    }

    const std::unique_ptr<EncodeState> state(new EncodeState());
    state->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, png_silent_error,
                                         png_silent_warning);
    if (state->png == nullptr) {
        throw ProjectIOError("PNG: cannot allocate an encoder");
    }
    state->info = png_create_info_struct(state->png);
    if (state->info == nullptr) {
        throw ProjectIOError("PNG: cannot allocate image info");
    }
    if (setjmp(png_jmpbuf(state->png)) != 0) {
        throw ProjectIOError("PNG: failed to encode");
    }

    png_set_write_fn(state->png, &state->out, png_mem_write, png_mem_flush);

    const int colour_type = channels == 1   ? PNG_COLOR_TYPE_GRAY
                            : channels == 3 ? PNG_COLOR_TYPE_RGB
                                            : PNG_COLOR_TYPE_RGB_ALPHA;
    png_set_IHDR(state->png, state->info, static_cast<png_uint_32>(image.width()),
                 static_cast<png_uint_32>(image.height()), 8, colour_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(state->png, state->info);

    // Rows are already tightly packed in the Image, so the row pointers hand
    // libpng the source buffer directly with no per-row copy.
    const std::size_t row_bytes =
        static_cast<std::size_t>(image.width()) * static_cast<std::size_t>(channels);
    state->rows.resize(static_cast<std::size_t>(image.height()));
    for (int y = 0; y < image.height(); ++y) {
        state->rows[static_cast<std::size_t>(y)] =
            const_cast<png_bytep>(image.data()) + static_cast<std::size_t>(y) * row_bytes;
    }
    png_write_image(state->png, state->rows.data());
    png_write_end(state->png, nullptr);

    // libpng can no longer longjmp past this point.
    return std::move(state->out);
}

}  // namespace alphashape::image_detail
