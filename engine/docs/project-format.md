# Project container format

The on-disk project is produced by `common/io/project_io.{hpp,cpp}`.

## Directory form (default)

    myproject.psx              text manifest
    myproject.files/
        project.zip            doc.xml + binary chunk sidecars
        cache/                 scratch: undo spill, GeoTIFF codec files
        lock                   present only while a writer holds the project

`myproject.psx` is a small text manifest:

    alphashape-project <format-version>
    files=myproject.files/project.zip

`project.zip` holds `doc.xml` at its root and every chunk payload sidecar under
`chunks/<chunk_id>/` (tie points, depth maps, point clouds, models, DEM,
orthomosaic, masks, shapes, trajectories, frames). The format version lives in
`<document version="...">` in `doc.xml`; see `common/io/doc_xml.hpp`.

A save writes `project.zip` through a temp file and rename, so a failed save
never truncates the previous project. Two saves of the same document are
byte-identical, and every text part is LF-only on every platform.

## Single-file form (.psz)

A path ending in `.psz` writes one zip containing exactly the entries
`project.zip` contains: `doc.xml` at the root plus the same sidecars. Loading a
`.psz` parses that archive. There is no nested `project.zip`, and the `.files`
directory is not created for this form.

The archive is held in memory on both save and load. A `.psz` is therefore
bounded by RAM during I/O; the directory form has the same in-memory archive,
but its bytes come from a file the caller can still stream externally.

## Frame paths

`doc.xml` (and the `frames.bin` sidecar) store each frame's `path`. Paths are
stored relative to the project file when the photo is on the same drive/root,
and absolute otherwise. All stored separators are `/`.

A save with `absolute_paths=true` stores every path absolutely, including one
that was relative in memory (resolved against the project directory first).

`Project/AbsolutePaths` is written as `true` only for that non-default choice;
its absence means relative. It is consumed on load and does not become part of
the document's own metadata.

On load, a relative path is resolved against the directory holding the project
file. A missing photo is a warning in `ReadReport`, never an error; the frame
keeps the stored text so the project remains loadable and can be relinked.
`doc.xml` files written before this rule are absolute (or relative to the
process's working directory) and keep loading as before.

## Cache

`Document::cache_dir()` is `<base>.files/cache` for a saved project and empty for
an unsaved one. It is derived from the document path; it is never written into
`doc.xml`. Undo spill files and the file-based GeoTIFF codec use it when it is
available and fall back to the platform temporary directory otherwise.

The directory is removed when the last `Document` naming that project is
destroyed. Only the `cache` subdirectory is touched.

## Writer lock

`ProjectLock::acquire(project_path)` writes `<base>.files/lock` for the
directory form and `<path>.lock` for a `.psz` (a single-file project owns no
`.files` directory, so no directory is created for it), with the pid, host name,
start time and a random token. A second writer raises `ProjectIOError` naming
the holder. When the lock cannot be created for permission or
read-only-filesystem reasons, `acquire` raises `ProjectReadOnlyError` instead:
that is not a second writer.

A lock is stale when its host is this machine and the pid is no longer running,
or when it came from another host and is older than `max_age_seconds`
(default 24 h). A stale lock is replaced, but never by deleting the path: it is
renamed aside (`lock.stale.<token>`), re-read, and compared with the record
that was judged stale. A different record (a writer that won the race) is put
back and reported as the holder, so two processes breaking the same stale lock
cannot remove the fresh lock one of them just created. The lock is released on
save-close (RAII) and read-only opens never take it. `ProjectLock::release()`
removes the file only when its token still matches, so a stale-break by another
process is not undone.
