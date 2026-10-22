#include <csetjmp>
#include <cstring>
#include <memory>
#include <vector>

#include <jerror.h>
#include <jpeglib.h>

#include "base/utils/error.hpp"
#include "image/io/codec.hpp"

namespace alphashape::image_detail {

namespace {

// libjpeg's default error_exit calls exit(), which we must never allow. Every
// decompressor here is built on this error manager instead: error_exit records
// the library's own formatted message and longjmps out of libjpeg, which the
// caller turns into a thrown Error. emit_message is neutered too, so a warning
// can never print to stderr during a test run — but it *counts* corrupt-data
// warnings, because those are how libjpeg reports damage it otherwise papers
// over (a truncated scan is filled with grey and reported only as
// JWRN_JPEG_EOF). A decode that saw any such warning is a failed decode.
struct JpegError {
    jpeg_error_mgr base;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
    int corruptions;
};

void jpeg_error_exit(j_common_ptr cinfo) {
    JpegError* err = reinterpret_cast<JpegError*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, err->message);
    std::longjmp(err->jump, 1);
}

void jpeg_emit_message(j_common_ptr cinfo, int msg_level) {
    // msg_level < 0 marks a warning. Only the codes that mean the compressed
    // data is actually damaged count as a failure: the others (an unknown Adobe
    // transform, a JFIF revision we do not know) are notes about metadata, and
    // treating them as corruption would reject perfectly decodable files.
    if (msg_level < 0) {
        // msg_code is an enum, so the list has to name only codes the header
        // declares unconditionally; JWRN_ARITH_BAD_CODE and friends are
        // conditionally defined and cannot be referenced portably.
        switch (cinfo->err->msg_code) {
            case JWRN_JPEG_EOF:
            case JWRN_HIT_MARKER:
            case JWRN_MUST_RESYNC:
            case JWRN_HUFF_BAD_CODE:
            case JWRN_BOGUS_PROGRESSION:
                ++reinterpret_cast<JpegError*>(cinfo->err)->corruptions;
                break;
            default:
                break;
        }
    }
}

// Everything the decode touches lives behind this one heap pointer. It is
// created before the setjmp and never reassigned, so a longjmp out of libjpeg
// leaves no automatic object with an indeterminate value behind: the throw
// that follows unwinds normally and the destructor frees everything.
struct JpegState {
    JpegState() {
        std::memset(&cinfo, 0, sizeof(cinfo));
        err.message[0] = '\0';
        err.corruptions = 0;
        cinfo.err = jpeg_std_error(&err.base);
        err.base.error_exit = jpeg_error_exit;
        err.base.emit_message = jpeg_emit_message;
    }

    ~JpegState() {
        if (created) {
            jpeg_destroy_decompress(&cinfo);
        }
    }

    jpeg_decompress_struct cinfo{};
    JpegError err{};
    bool created = false;
    std::vector<std::uint8_t> pixels;
};

/// Allocates the decompressor, reads the header and refuses anything outside
/// the supported contract. The caller's setjmp is already armed.
void create_and_read_header(JpegState& state, const std::uint8_t* data, std::size_t size) {
    jpeg_create_decompress(&state.cinfo);
    state.created = true;
    // jpeg_mem_src takes an unsigned long length; a JPEG bigger than that is
    // not a file we will ever meet.
    jpeg_mem_src(&state.cinfo, data, static_cast<unsigned long>(size));
    if (jpeg_read_header(&state.cinfo, TRUE) != JPEG_HEADER_OK) {
        throw ProjectIOError("JPEG: no SOF header found");
    }
    const jpeg_decompress_struct& cinfo = state.cinfo;
    if (cinfo.data_precision != 8) {
        throw ProjectIOError("JPEG: only 8-bit sample precision is supported (file is " +
                             std::to_string(static_cast<int>(cinfo.data_precision)) + "-bit)");
    }
    if (cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK) {
        throw ProjectIOError("JPEG: CMYK/YCCK colour is not supported");
    }
    if (cinfo.jpeg_color_space != JCS_GRAYSCALE && cinfo.jpeg_color_space != JCS_RGB &&
        cinfo.jpeg_color_space != JCS_YCbCr) {
        throw ProjectIOError("JPEG: unsupported colour space");
    }
}

}  // namespace

ImageInfo jpeg_info(const std::uint8_t* data, std::size_t size) {
    const std::unique_ptr<JpegState> state(new JpegState());
    if (setjmp(state->err.jump) != 0) {
        throw ProjectIOError("JPEG: " + std::string(state->err.message));
    }
    create_and_read_header(*state, data, size);

    ImageInfo info;
    info.format = ImageFormat::Jpeg;
    info.width = static_cast<int>(state->cinfo.image_width);
    info.height = static_cast<int>(state->cinfo.image_height);
    info.channels = state->cinfo.jpeg_color_space == JCS_GRAYSCALE ? 1 : 3;
    info.bit_depth = static_cast<int>(state->cinfo.data_precision);
    return info;
}

Image jpeg_decode(const std::uint8_t* data, std::size_t size) {
    const std::unique_ptr<JpegState> state(new JpegState());
    if (setjmp(state->err.jump) != 0) {
        throw ProjectIOError("JPEG: " + std::string(state->err.message));
    }
    create_and_read_header(*state, data, size);

    jpeg_decompress_struct& cinfo = state->cinfo;
    const bool grey = cinfo.jpeg_color_space == JCS_GRAYSCALE;
    cinfo.out_color_space = grey ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_start_decompress(&cinfo);

    const int width = static_cast<int>(cinfo.output_width);
    const int height = static_cast<int>(cinfo.output_height);
    const int components = static_cast<int>(cinfo.output_components);
    if (components != (grey ? 1 : 3)) {
        throw ProjectIOError("JPEG: unexpected output component count");
    }
    // A JPEG's SOF can claim any 16-bit dimension; guard before allocating.
    check_decodable(static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(height),
                    static_cast<std::uint64_t>(components), "JPEG");

    state->pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                         static_cast<std::size_t>(components));
    const std::size_t row_stride =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(components);

    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row =
            state->pixels.data() + static_cast<std::size_t>(cinfo.output_scanline) * row_stride;
        JSAMPROW rows[1] = {row};
        if (jpeg_read_scanlines(&cinfo, rows, 1) != 1) {
            throw ProjectIOError("JPEG: truncated scanline data");
        }
    }
    jpeg_finish_decompress(&cinfo);

    // libjpeg reports a truncated entropy-coded stream as a warning and fills
    // the missing rows with grey rather than failing; we treat any corrupt-data
    // warning as a decode failure so the caller never receives a half-image.
    if (state->err.corruptions > 0) {
        throw ProjectIOError("JPEG: corrupt or truncated data");
    }

    Image result(width, height, components);
    std::memcpy(result.data(), state->pixels.data(), state->pixels.size());
    return result;
}

}  // namespace alphashape::image_detail
