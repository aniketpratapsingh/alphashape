#!/usr/bin/env python3
"""Make an alphashape.app bundle self-contained.

Walks every Mach-O file in the bundle, copies each non-system dylib it references into
Contents/Frameworks, rewrites install names to @rpath/<name>, replaces absolute rpaths with a
bundle-relative one, then audits the result: every reference must be a system library or resolve
inside the bundle. Qt itself is handled earlier by macdeployqt; this closes whatever it leaves
(Ceres, PROJ, glog, gflags, jpeg, png, tiff, libomp, the Vulkan loader, MoltenVK, ...).

Also prints (and with --min-os-out writes) the highest macOS deployment target of any Mach-O in
the bundle, which becomes LSMinimumSystemVersion.

Usage: bundle_libs.py <alphashape.app> [--min-os-out FILE]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

SYSTEM_PREFIXES = ("/usr/lib/", "/System/")
MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
}


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def is_macho(path):
    if os.path.islink(path) or not os.path.isfile(path):
        return False
    with open(path, "rb") as handle:
        return handle.read(4) in MACHO_MAGICS


def macho_files(app):
    found = []
    for root, _dirs, files in os.walk(app):
        for name in files:
            path = os.path.join(root, name)
            if is_macho(path):
                found.append(path)
    return sorted(found)


def deps_of(path):
    """Install names referenced by `path` (its own id is excluded)."""
    lines = run(["otool", "-L", path]).splitlines()[1:]
    own_id = id_of(path)
    deps = []
    for line in lines:
        match = re.match(r"\s+(\S.*?) \(compatibility", line)
        if match and match.group(1) != own_id:
            deps.append(match.group(1))
    return deps


def id_of(path):
    """Install-name id of a dylib/framework binary, or '' for executables and bundles."""
    # Output is "<path>:" (or "<path> (architecture arm64):") followed by the id, if there is one.
    lines = [l.strip() for l in run(["otool", "-D", path]).splitlines()]
    ids = [l for l in lines if l and not l.endswith(":")]
    return ids[-1] if ids else ""


def rpaths_of(path):
    out = run(["otool", "-l", path]).splitlines()
    rpaths = []
    for i, line in enumerate(out):
        if line.strip() == "cmd LC_RPATH":
            match = re.match(r"\s+path (.*) \(offset", out[i + 2])
            if match:
                rpaths.append(match.group(1))
    return rpaths


def min_os_of(path):
    best = (0, 0)
    out = run(["otool", "-l", path]).splitlines()
    for i, line in enumerate(out):
        stripped = line.strip()
        key = None
        if stripped == "cmd LC_BUILD_VERSION":
            key = "minos"
        elif stripped == "cmd LC_VERSION_MIN_MACOSX":
            key = "version"
        if key:
            for follow in out[i + 1:i + 8]:
                parts = follow.split()
                if len(parts) == 2 and parts[0] == key:
                    nums = parts[1].split(".")
                    best = max(best, (int(nums[0]), int(nums[1]) if len(nums) > 1 else 0))
                    break
    return best


BREW_ROOTS = ("/opt/homebrew", "/usr/local")


def find_in_homebrew(name):
    """Locate a library by file name under a Homebrew prefix.

    macdeployqt strips the rpaths of what it copies, so a library such as libgfortran arrives
    with `@rpath/libgcc_s.1.1.dylib` and nowhere to look; Homebrew keeps those in versioned
    keg directories.
    """
    for root in BREW_ROOTS:
        for pattern in (f"{root}/lib/{name}", f"{root}/lib/gcc/current/{name}",
                        f"{root}/opt/*/lib/{name}", f"{root}/opt/*/lib/gcc/current/{name}",
                        f"{root}/Cellar/*/*/lib/{name}", f"{root}/Cellar/*/*/lib/gcc/*/{name}"):
            hits = sorted(glob.glob(pattern))
            if hits:
                return hits[0]
    return None


def qt_lib_dir():
    root = os.environ.get("QT_ROOT_DIR", "")
    return os.path.join(root, "lib") if root else ""


class Bundle:
    def __init__(self, app):
        self.app = os.path.realpath(app)
        self.frameworks = os.path.join(self.app, "Contents", "Frameworks")
        os.makedirs(self.frameworks, exist_ok=True)

    def inside(self, path):
        real = os.path.realpath(path)
        return real == self.app or real.startswith(self.app + os.sep)

    def expand(self, ref, owner, rpaths):
        """Resolve one reference to existing file paths (candidates), best first."""
        owner_dir = os.path.dirname(owner)
        exe_dir = os.path.join(self.app, "Contents", "MacOS")
        if ref.startswith("@loader_path/"):
            return [os.path.join(owner_dir, ref[len("@loader_path/"):])]
        if ref.startswith("@executable_path/"):
            return [os.path.join(exe_dir, ref[len("@executable_path/"):])]
        if ref.startswith("@rpath/"):
            tail = ref[len("@rpath/"):]
            candidates = []
            for rpath in rpaths:
                base = rpath.replace("@loader_path", owner_dir).replace("@executable_path", exe_dir)
                candidates.append(os.path.join(base, tail))
            return candidates
        return [ref]

    def resolve(self, ref, owner, rpaths):
        for candidate in self.expand(ref, owner, rpaths):
            if os.path.exists(candidate):
                return os.path.normpath(candidate)
        return None

    def copy_in(self, source):
        name = os.path.basename(source)
        if ".framework/" in source:
            sys.exit(f"error: framework outside the bundle was not deployed by macdeployqt: {source}")
        dest = os.path.join(self.frameworks, name)
        if not os.path.exists(dest):
            shutil.copy2(os.path.realpath(source), dest)
            os.chmod(dest, 0o755)
            run(["install_name_tool", "-id", f"@rpath/{name}", dest])
            print(f"  bundled {name}  <-  {source}")
        return dest

    @staticmethod
    def qt_framework_ref(ref):
        """(framework name, path inside it) when `ref` points into a Qt framework, else None.

        Qt plugins copied out of the Qt install by build-app.sh still use both `@rpath/...` and
        absolute references into that install, so both spellings have to be recognised."""
        match = re.search(r"(Qt\w+\.framework)/", ref)
        return (match.group(1), ref[match.end():]) if match else None

    def deploy_qt_framework(self, ref, resolved=None):
        """Copy a Qt framework that is missing from the bundle but referenced by something
        already in it (for example QtSvg, required by the SVG icon-engine plugin).

        `resolved` is the reference already expanded to a path outside the bundle, which is
        where the framework content is copied from; QT_ROOT_DIR is the fallback source."""
        parsed = self.qt_framework_ref(ref)
        if not parsed:
            return None
        name, tail = parsed
        source = None
        if resolved:
            source = resolved[:resolved.index(name) + len(name)]
        if not source or not os.path.isdir(source):
            source = os.path.join(qt_lib_dir(), name)
        dest = os.path.join(self.frameworks, name)
        if not os.path.isdir(source):
            return None
        if not os.path.exists(dest):
            shutil.copytree(source, dest, symlinks=True,
                            ignore=shutil.ignore_patterns("Headers", "*.prl"))
            print(f"  deployed {name}  <-  {source}")
        return os.path.join(dest, tail)

    def fix_rpaths(self, path):
        wanted = "@loader_path/" + os.path.relpath(self.frameworks, os.path.dirname(path))
        current = rpaths_of(path)
        for rpath in current:
            if not rpath.startswith("@"):
                run(["install_name_tool", "-delete_rpath", rpath, path])
        if wanted not in current:
            run(["install_name_tool", "-add_rpath", wanted, path])

    def process(self):
        done = set()
        while True:
            pending = [p for p in macho_files(self.app) if p not in done]
            if not pending:
                break
            for path in pending:
                done.add(path)
                rpaths = rpaths_of(path)
                for ref in deps_of(path):
                    if ref.startswith(SYSTEM_PREFIXES):
                        continue
                    source = self.resolve(ref, path, rpaths)
                    if source is None:
                        # An @rpath reference may already point at a bundled copy.
                        if os.path.exists(os.path.join(self.frameworks, os.path.basename(ref))):
                            source = os.path.join(self.frameworks, os.path.basename(ref))
                        else:
                            source = (self.deploy_qt_framework(ref)
                                      or find_in_homebrew(os.path.basename(ref)))
                        if source is None:
                            sys.exit(f"error: cannot resolve {ref} (needed by {path})")
                    framework = self.qt_framework_ref(ref)
                    if framework:
                        # A Qt framework binary keeps its framework-relative install name, so
                        # the framework finds its own QtCore; copy_in would flatten it.
                        name, tail = framework
                        if not self.inside(source):
                            source = self.deploy_qt_framework(ref, source)
                            if source is None:
                                sys.exit(f"error: cannot deploy {ref} (needed by {path})")
                        canonical = f"@rpath/{name}/{tail}"
                        if ref != canonical:
                            run(["install_name_tool", "-change", ref, canonical, path])
                        continue
                    if self.inside(source):
                        # Already in the bundle, but the spelling matters: `@executable_path`
                        # is the Python interpreter for a .so loaded into it and `./lib/x.dylib`
                        # is silently used by nothing on a modern dyld. Canonicalise to @rpath.
                        canonical = "@rpath/" + os.path.basename(source)
                        if ref != canonical:
                            run(["install_name_tool", "-change", ref, canonical, path])
                        continue
                    dest = self.copy_in(source)
                    run(["install_name_tool", "-change", ref,
                         "@rpath/" + os.path.basename(dest), path])
                self.fix_rpaths(path)
                own_id = id_of(path)
                if own_id and not own_id.startswith("@rpath/") and not self.qt_framework_ref(own_id):
                    # macdeployqt leaves Homebrew install names and the @executable_path spelling
                    # on what it copies. The latter means "next to the process", which is the
                    # Python interpreter for the module in Resources/python.
                    run(["install_name_tool", "-id", "@rpath/" + os.path.basename(own_id), path])

    def audit(self):
        bad = []
        files = macho_files(self.app)
        for path in files:
            rpaths = rpaths_of(path)
            if id_of(path) and not id_of(path).startswith("@rpath/") \
                    and not self.qt_framework_ref(id_of(path)):
                bad.append(f"{path}: non-@rpath install name {id_of(path)}")
            for rpath in rpaths:
                if not rpath.startswith("@"):
                    bad.append(f"{path}: absolute rpath {rpath}")
            for ref in deps_of(path):
                if ref.startswith(SYSTEM_PREFIXES):
                    continue
                if ref.startswith("/"):
                    bad.append(f"{path}: absolute reference {ref}")
                    continue
                if ref.startswith("@executable_path/"):
                    # Only true for the two executables; a .so loaded into python3 or a Qt
                    # plugin loaded into the app would look for it next to the wrong binary.
                    bad.append(f"{path}: @executable_path reference {ref}")
                    continue
                source = self.resolve(ref, path, rpaths)
                if source is None or not self.inside(source):
                    bad.append(f"{path}: {ref} does not resolve inside the bundle")
        if bad:
            sys.exit("bundle audit failed:\n  " + "\n  ".join(bad))
        print(f"audit ok: {len(files)} Mach-O files, every reference is a system library or bundled")
        return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("app")
    parser.add_argument("--min-os-out")
    args = parser.parse_args()

    bundle = Bundle(args.app)
    bundle.process()
    files = bundle.audit()

    minimum = max((min_os_of(p) for p in files), default=(11, 0))
    text = f"{minimum[0]}.{minimum[1]}"
    print(f"minimum macOS version across the bundle: {text}")
    if args.min_os_out:
        with open(args.min_os_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
