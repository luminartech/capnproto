// TODO: copyright

#if __QNX__

#include "filesystem.h"
#include "debug.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/mman.h>
#include <errno.h>
#include <dirent.h>
#include <stdlib.h>
#include "vector.h"
#include "miniposix.h"
#include <algorithm>

#include <cassert>

namespace {

  std::string fs_join(std::string path, std::string child)
  {
    if (path.back() == '/')
    {
      return path + child;
    }

    return path + '/' + child;
  }
}

namespace kj {
namespace {

#define HIDDEN_PREFIX ".kj-tmp."
// Prefix for temp files which should be hidden when listing a directory.
//
// If you change this, make sure to update the unit test.

#define MAYBE_O_CLOEXEC O_CLOEXEC

// No `O_DIRECTORY` on QNX 7
#define MAYBE_O_DIRECTORY 0

static void setCloexec(int fd) KJ_UNUSED;
static void setCloexec(int fd) {
  // Set the O_CLOEXEC flag on the given fd.
  //
  // We try to avoid the need to call this by taking advantage of syscall flags that set it
  // atomically on new file descriptors. Unfortunately some platforms do not support such syscalls.

  KJ_SYSCALL_HANDLE_ERRORS(ioctl(fd, FIOCLEX)) {
    case EINVAL:
    case EOPNOTSUPP:
      break;
    default:
      KJ_FAIL_SYSCALL("ioctl(fd, FIOCLEX)", error) { break; }
      break;
  } else {
    // success
    return;
  }
}

static Date toKjDate(struct timespec tv) {
  return tv.tv_sec * SECONDS + tv.tv_nsec * NANOSECONDS + UNIX_EPOCH;
}

static FsNode::Type modeToType(mode_t mode) {
  switch (mode & S_IFMT) {
    case S_IFREG : return FsNode::Type::FILE;
    case S_IFDIR : return FsNode::Type::DIRECTORY;
    case S_IFLNK : return FsNode::Type::SYMLINK;
    case S_IFBLK : return FsNode::Type::BLOCK_DEVICE;
    case S_IFCHR : return FsNode::Type::CHARACTER_DEVICE;
    case S_IFIFO : return FsNode::Type::NAMED_PIPE;
    case S_IFSOCK: return FsNode::Type::SOCKET;
    default: return FsNode::Type::OTHER;
  }
}

static FsNode::Metadata statToMetadata(struct stat& stats) {
  // Probably st_ino and st_dev are usually under 32 bits, so mix by rotating st_dev left 32 bits
  // and XOR.
  uint64_t d = stats.st_dev;
  uint64_t hash = ((d << 32) | (d >> 32)) ^ stats.st_ino;

  return FsNode::Metadata {
    modeToType(stats.st_mode),
    implicitCast<uint64_t>(stats.st_size),
    implicitCast<uint64_t>(stats.st_blocks * 512u),
    toKjDate(stats.st_mtim),
    implicitCast<uint>(stats.st_nlink),
    hash
  };
}

// TODO: update all of these to no longer take fd params
static bool rmrf(int fd, StringPtr path);

static void rmrfChildrenAndClose(int fd) {

  assert(false);
}

static bool rmrf(int fd, StringPtr path) {

  assert(false);

  return true;
}

struct MmapRange {
  uint64_t offset;
  uint64_t size;
};

static MmapRange getMmapRange(uint64_t offset, uint64_t size) {
  // Comes up with an offset and size to pass to mmap(), given an offset and size requested by
  // the caller, and considering the fact that mappings must start at a page boundary.
  //
  // The offset is rounded down to the nearest page boundary, and the size is increased to
  // compensate. Note that the endpoint of the mapping is *not* rounded up to a page boundary, as
  // mmap() does not actually require this, and it causes trouble on some systems (notably Cygwin).

#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE _SC_PAGE_SIZE
#endif
  static const uint64_t pageSize = sysconf(_SC_PAGESIZE);
  uint64_t pageMask = pageSize - 1;

  uint64_t realOffset = offset & ~pageMask;

  return { realOffset, offset + size - realOffset };
}

class MmapDisposer: public ArrayDisposer {
protected:
  void disposeImpl(void* firstElement, size_t elementSize, size_t elementCount,
                   size_t capacity, void (*destroyElement)(void*)) const {
    auto range = getMmapRange(reinterpret_cast<uintptr_t>(firstElement),
                              elementSize * elementCount);
    KJ_SYSCALL(munmap(reinterpret_cast<byte*>(range.offset), range.size)) { break; }
  }
};

constexpr MmapDisposer mmapDisposer = MmapDisposer();

class DiskHandle {
  // We need to implement each of ReadableFile, AppendableFile, File, ReadableDirectory, and
  // Directory for disk handles. There is a lot of implementation overlap between these, especially
  // stat(), sync(), etc. We can't have everything inherit from a common DiskFsNode that implements
  // these because then we get diamond inheritance which means we need to make all our inheritance
  // virtual which means downcasting requires RTTI which violates our goal of supporting compiling
  // with no RTTI. So instead we have the DiskHandle class which implements all the methods without
  // inheriting anything, and then we have DiskFile, DiskDirectory, etc. hold this and delegate to
  // it. Ugly, but works.

public:
  DiskHandle(AutoCloseFd&& fd): fd(kj::mv(fd)) {}

  // OsHandle ------------------------------------------------------------------

  AutoCloseFd clone() const {
    int fd2;
    KJ_SYSCALL_HANDLE_ERRORS(fd2 = fcntl(fd, F_DUPFD_CLOEXEC, 3)) {
      case EINVAL:
      case EOPNOTSUPP:
        // fall back
        break;
      default:
        KJ_FAIL_SYSCALL("fnctl(fd, F_DUPFD_CLOEXEC, 3)", error) { break; }
        break;
    } else {
      return AutoCloseFd(fd2, fd.get_path());
    }

    KJ_SYSCALL(fd2 = ::dup(fd));
    AutoCloseFd result(fd2, fd.get_path());
    setCloexec(result);
    return result;
  }

  int getFd() const {
    return fd.get();
  }

  std::string getFdPath() const {
    return fd.get_path();
  }

  // FsNode --------------------------------------------------------------------

  FsNode::Metadata stat() const {
    struct stat stats;
    KJ_SYSCALL(::fstat(fd, &stats));
    return statToMetadata(stats);
  }

  void sync() const {
    KJ_SYSCALL(fsync(fd));
  }

  void datasync() const {
    KJ_SYSCALL(fdatasync(fd));
  }

  // ReadableFile --------------------------------------------------------------

  size_t read(uint64_t offset, ArrayPtr<byte> buffer) const {
    // pread() probably never returns short reads unless it hits EOF. Unfortunately, though, per
    // spec we are not allowed to assume this.

    size_t total = 0;
    while (buffer.size() > 0) {
      ssize_t n;
      KJ_SYSCALL(n = pread(fd, buffer.begin(), buffer.size(), offset));
      if (n == 0) break;
      total += n;
      offset += n;
      buffer = buffer.slice(n, buffer.size());
    }
    return total;
  }

  Array<const byte> mmap(uint64_t offset, uint64_t size) const {
    auto range = getMmapRange(offset, size);
    const void* mapping = ::mmap(NULL, range.size, PROT_READ, MAP_SHARED, fd, range.offset);
    if (mapping == MAP_FAILED) {
      KJ_FAIL_SYSCALL("mmap", errno);
    }
    return Array<const byte>(reinterpret_cast<const byte*>(mapping) + (offset - range.offset),
                             size, mmapDisposer);
  }

  Array<byte> mmapPrivate(uint64_t offset, uint64_t size) const {
    auto range = getMmapRange(offset, size);
    void* mapping = ::mmap(NULL, range.size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, range.offset);
    if (mapping == MAP_FAILED) {
      KJ_FAIL_SYSCALL("mmap", errno);
    }
    return Array<byte>(reinterpret_cast<byte*>(mapping) + (offset - range.offset),
                       size, mmapDisposer);
  }

  // File ----------------------------------------------------------------------

  void write(uint64_t offset, ArrayPtr<const byte> data) const {
    // pwrite() probably never returns short writes unless there's no space left on disk.
    // Unfortunately, though, per spec we are not allowed to assume this.

    while (data.size() > 0) {
      ssize_t n;
      KJ_SYSCALL(n = pwrite(fd, data.begin(), data.size(), offset));
      KJ_ASSERT(n > 0, "pwrite() returned zero?");
      offset += n;
      data = data.slice(n, data.size());
    }
  }

  void zero(uint64_t offset, uint64_t size) const {
    static const byte ZEROS[4096] = { 0 };

    // Mac & Cygwin doesn't have pwritev().
    while (size > sizeof(ZEROS)) {
      write(offset, ZEROS);
      size -= sizeof(ZEROS);
      offset += sizeof(ZEROS);
    }
    write(offset, kj::arrayPtr(ZEROS, size));
  }

  void truncate(uint64_t size) const {
    KJ_SYSCALL(ftruncate(fd, size));
  }

  class WritableFileMappingImpl final: public WritableFileMapping {
  public:
    WritableFileMappingImpl(Array<byte> bytes): bytes(kj::mv(bytes)) {}

    ArrayPtr<byte> get() const override {
      // const_cast OK because WritableFileMapping does indeed provide a writable view despite
      // being const itself.
      return arrayPtr(const_cast<byte*>(bytes.begin()), bytes.size());
    }

    void changed(ArrayPtr<byte> slice) const override {
      KJ_REQUIRE(slice.begin() >= bytes.begin() && slice.end() <= bytes.end(),
                 "byte range is not part of this mapping");

      // msync() requires page-alignment, apparently, so use getMmapRange() to accomplish that.
      auto range = getMmapRange(reinterpret_cast<uintptr_t>(slice.begin()), slice.size());
      KJ_SYSCALL(msync(reinterpret_cast<void*>(range.offset), range.size, MS_ASYNC));
    }

    void sync(ArrayPtr<byte> slice) const override {
      KJ_REQUIRE(slice.begin() >= bytes.begin() && slice.end() <= bytes.end(),
                 "byte range is not part of this mapping");

      // msync() requires page-alignment, apparently, so use getMmapRange() to accomplish that.
      auto range = getMmapRange(reinterpret_cast<uintptr_t>(slice.begin()), slice.size());
      KJ_SYSCALL(msync(reinterpret_cast<void*>(range.offset), range.size, MS_SYNC));
    }

  private:
    Array<byte> bytes;
  };

  Own<const WritableFileMapping> mmapWritable(uint64_t offset, uint64_t size) const {
    auto range = getMmapRange(offset, size);
    void* mapping = ::mmap(NULL, range.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, range.offset);
    if (mapping == MAP_FAILED) {
      KJ_FAIL_SYSCALL("mmap", errno);
    }
    auto array = Array<byte>(reinterpret_cast<byte*>(mapping) + (offset - range.offset),
                             size, mmapDisposer);
    return heap<WritableFileMappingImpl>(kj::mv(array));
  }

  size_t copyChunk(uint64_t offset, int fromFd, uint64_t fromOffset, uint64_t size) const {
    // Copies a range of bytes from `fromFd` to this file in the most efficient way possible for
    // the OS. Only returns less than `size` if EOF. Does not account for holes.

    uint64_t total = 0;
    while (size > 0) {
      byte buffer[4096];
      ssize_t n;
      KJ_SYSCALL(n = pread(fromFd, buffer, kj::min(sizeof(buffer), size), fromOffset));
      if (n == 0) break;
      write(offset, arrayPtr(buffer, n));
      fromOffset += n;
      offset += n;
      total += n;
      size -= n;
    }
    return total;
  }

  kj::Maybe<size_t> copy(uint64_t offset, const ReadableFile& from,
                         uint64_t fromOffset, uint64_t size) const {
    KJ_IF_MAYBE(otherFd, from.getFd()) {

      off_t toPos = offset;
      off_t fromPos = fromOffset;
      off_t end = size == kj::maxValue ? off_t(kj::maxValue) : off_t(fromOffset + size);

      for (;;) {
        // Handle data.
        {
          // Find out how much data there is before the next hole.
          off_t nextHole;
          // SEEK_HOLE not supported. Assume no holes.
          nextHole = end;

          // Copy the next chunk of data.
          off_t copyTo = kj::min(end, nextHole);
          size_t amount = copyTo - fromPos;
          if (amount > 0) {
            size_t n = copyChunk(toPos, *otherFd, fromPos, amount);
            fromPos += n;
            toPos += n;

            if (n < amount) {
              return fromPos - fromOffset;
            }
          }

          if (fromPos == end) {
            return fromPos - fromOffset;
          }
        }
      }
    }

    // Indicates caller should call File::copy() default implementation.
    return nullptr;
  }

  // ReadableDirectory ---------------------------------------------------------

  template <typename Func>
  auto list(bool needTypes, Func&& func) const
      -> Array<Decay<decltype(func(instance<StringPtr>(), instance<FsNode::Type>()))>> {
    // Seek to start of directory.
    KJ_SYSCALL(lseek(fd, 0, SEEK_SET));

    const auto p = fd.get_path().c_str();
    DIR* dir = opendir(p);

    KJ_DEFER(closedir(dir));
    typedef Decay<decltype(func(instance<StringPtr>(), instance<FsNode::Type>()))> Entry;
    kj::Vector<Entry> entries;

    for (;;) {
      errno = 0;
      struct dirent* entry = readdir(dir);
      if (entry == nullptr) {
        int error = errno;
        if (error == 0) {
          break;
        } else {
          KJ_FAIL_SYSCALL("readdir", error);
        }
      }

      kj::StringPtr name = entry->d_name;
      if (name != "." && name != ".." && !name.startsWith(HIDDEN_PREFIX)) {
          if (needTypes) {
            // Unknown type. Fall back to stat.
            struct stat stats;
            KJ_SYSCALL(fstatat(fd, name.cStr(), &stats, AT_SYMLINK_NOFOLLOW));
            entries.add(func(name, modeToType(stats.st_mode)));
          } else {
            entries.add(func(name, FsNode::Type::OTHER));
          }
      }
    }

    auto result = entries.releaseAsArray();
    std::sort(result.begin(), result.end());
    return result;
  }

  Array<String> listNames() const {
    return list(false, [](StringPtr name, FsNode::Type type) { return heapString(name); });
  }

  Array<ReadableDirectory::Entry> listEntries() const {
    return list(true, [](StringPtr name, FsNode::Type type) {
      return ReadableDirectory::Entry { type, heapString(name), };
    });
  }

  bool exists(PathPtr path) const {
    const auto p = fs_join(fd.get_path(), path.toString().cStr());
    KJ_SYSCALL_HANDLE_ERRORS(access(p.c_str(), F_OK)) {
      case ENOENT:
      case ENOTDIR:
        return false;
      default:
        KJ_FAIL_SYSCALL("access(path, mode)", error, path) { return false; }
    }
    return true;
  }

  Maybe<FsNode::Metadata> tryLstat(PathPtr path) const {
    struct stat stats;
    KJ_SYSCALL_HANDLE_ERRORS(fstatat(fd, path.toString().cStr(), &stats, AT_SYMLINK_NOFOLLOW)) {
      case ENOENT:
      case ENOTDIR:
        return nullptr;
      default:
        KJ_FAIL_SYSCALL("faccessat(fd, path)", error, path) { return nullptr; }
    }
    return statToMetadata(stats);
  }

  Maybe<Own<const ReadableFile>> tryOpenFile(PathPtr path) const {
    int newFd;
    const auto p = fs_join(fd.get_path(), path.toString().cStr());
    KJ_SYSCALL_HANDLE_ERRORS(newFd = open(
                               p.c_str(), O_RDONLY | MAYBE_O_CLOEXEC)) {
      case ENOENT:
      case ENOTDIR:
        return nullptr;
      default:
        KJ_FAIL_SYSCALL("open(path, O_RDONLY)", error, p) { return nullptr; }
    }

    kj::AutoCloseFd result(newFd, p);

    return newDiskReadableFile(kj::mv(result));
  }

  Maybe<AutoCloseFd> tryOpenSubdirInternal(PathPtr path) const {
    int newFd;
    const auto p = fs_join(fd.get_path(), path.toString().cStr());
    KJ_SYSCALL_HANDLE_ERRORS(newFd = open(
                               p.c_str(), O_RDONLY | MAYBE_O_CLOEXEC | MAYBE_O_DIRECTORY)) {
      case ENOENT:
        return nullptr;
      case ENOTDIR:
        // Could mean that a parent is not a directory, which we treat as "doesn't exist".
        // Could also mean that the specified file is not a directory, which should throw.
        // Check using exists().
        if (!exists(path)) {
          return nullptr;
        }
        // fallthrough
      default:
        KJ_FAIL_SYSCALL("open(path, O_DIRECTORY)", error, p) { return nullptr; }
    }

    kj::AutoCloseFd result(newFd, p);

    return kj::mv(result);
  }

  Maybe<Own<const ReadableDirectory>> tryOpenSubdir(PathPtr path) const {
    return tryOpenSubdirInternal(path).map(newDiskReadableDirectory);
  }

  Maybe<String> tryReadlink(PathPtr path) const {
    const auto p = fs_join(fd.get_path(), path.toString().cStr());
    size_t trySize = 256;
    for (;;) {
      KJ_STACK_ARRAY(char, buf, trySize, 256, 4096);
      ssize_t n = readlink(p.c_str(), buf.begin(), buf.size());
      if (n < 0) {
        int error = errno;
        switch (error) {
          case EINTR:
            continue;
          case ENOENT:
          case ENOTDIR:
          case EINVAL:    // not a link
            return nullptr;
          default:
            KJ_FAIL_SYSCALL("readlink(path)", error, p) { return nullptr; }
        }
      }

      if (n >= buf.size()) {
        // Didn't give it enough space. Better retry with a bigger buffer.
        trySize *= 2;
        continue;
      }

      return heapString(buf.begin(), n);
    }
  }

  // Directory -----------------------------------------------------------------

  bool tryMkdir(PathPtr path, WriteMode mode, bool noThrow) const {
    // Internal function to make a directory.

    auto filename = path.toString();
    mode_t acl = has(mode, WriteMode::PRIVATE) ? 0700 : 0777;

    const auto p = fs_join(fd.get_path(), filename.cStr());
    KJ_SYSCALL_HANDLE_ERRORS(mkdir(p.c_str(), acl)) {
      case EEXIST: {
        // Apparently this path exists.
        if (!has(mode, WriteMode::MODIFY)) {
          // Require exclusive create.
          return false;
        }

        // MODIFY is allowed, so we just need to check whether the existing entry is a directory.
        struct stat stats;
        KJ_SYSCALL_HANDLE_ERRORS(fstatat(fd, filename.cStr(), &stats, 0)) {
          default:
            // mkdir() says EEXIST but we can't stat it. Maybe it's a dangling link, or maybe
            // we can't access it for some reason. Assume failure.
            //
            // TODO(someday): Maybe we should be creating the directory at the target of the
            //   link?
            goto failed;
        }
        return (stats.st_mode & S_IFMT) == S_IFDIR;
      }
      case ENOENT:
        if (has(mode, WriteMode::CREATE_PARENT) && path.size() > 0 &&
            tryMkdir(path.parent(), WriteMode::CREATE | WriteMode::MODIFY |
                                    WriteMode::CREATE_PARENT, true)) {
          // Retry, but make sure we don't try to create the parent again.
          return tryMkdir(path, mode - WriteMode::CREATE_PARENT, noThrow);
        } else {
          goto failed;
        }
      default:
      failed:
        if (noThrow) {
          // Caller requested no throwing.
          return false;
        } else {
          KJ_FAIL_SYSCALL("mkdir(path)", error, path);
        }
    }

    return true;
  }

  kj::Maybe<String> createNamedTemporary(
      PathPtr finalName, WriteMode mode, Function<int(StringPtr)> tryCreate) const {
    // Create a temporary file which will eventually replace `finalName`.
    //
    // Calls `tryCreate` to actually create the temporary, passing in the desired path. tryCreate()
    // is expected to behave like a syscall, returning a negative value and setting `errno` on
    // error. tryCreate() MUST fail with EEXIST if the path exists -- this is not checked in
    // advance, since it needs to be checked atomically. In the case of EEXIST, tryCreate() will
    // be called again with a new path.
    //
    // Returns the temporary path that succeeded. Only returns nullptr if there was an exception
    // but we're compiled with -fno-exceptions.

    if (finalName.size() == 0) {
      KJ_FAIL_REQUIRE("can't replace self") { break; }
      return nullptr;
    }

    static uint counter = 0;
    static const pid_t pid = getpid();
    String pathPrefix;
    if (finalName.size() > 1) {
      pathPrefix = kj::str(finalName.parent(), '/');
    }
    auto path = kj::str(pathPrefix, HIDDEN_PREFIX, pid, '.', counter++, '.',
                        finalName.basename()[0], ".partial");

    KJ_SYSCALL_HANDLE_ERRORS(tryCreate(path)) {
      case EEXIST:
        return createNamedTemporary(finalName, mode, kj::mv(tryCreate));
      case ENOENT:
        if (has(mode, WriteMode::CREATE_PARENT) && finalName.size() > 1 &&
            tryMkdir(finalName.parent(), WriteMode::CREATE | WriteMode::MODIFY |
                                         WriteMode::CREATE_PARENT, true)) {
          // Retry, but make sure we don't try to create the parent again.
          mode = mode - WriteMode::CREATE_PARENT;
          return createNamedTemporary(finalName, mode, kj::mv(tryCreate));
        }
        // fallthrough
      default:
        KJ_FAIL_SYSCALL("create(path)", error, path) { break; }
        return nullptr;
    }

    return kj::mv(path);
  }

  bool tryReplaceNode(PathPtr path, WriteMode mode, Function<int(StringPtr)> tryCreate) const {
    // Replaces the given path with an object created by calling tryCreate().
    //
    // tryCreate() must behave like a syscall which creates the node at the path passed to it,
    // returning a negative value on error. If the path passed to tryCreate already exists, it
    // MUST fail with EEXIST.
    //
    // When `mode` includes MODIFY, replaceNode() reacts to EEXIST by creating the node in a
    // temporary location and then rename()ing it into place.

    if (path.size() == 0) {
      KJ_FAIL_REQUIRE("can't replace self") { return false; }
    }

    auto filename = path.toString();

    if (has(mode, WriteMode::CREATE)) {
      // First try just cerating the node in-place.
      KJ_SYSCALL_HANDLE_ERRORS(tryCreate(filename)) {
        case EEXIST:
          // Target exists.
          if (has(mode, WriteMode::MODIFY)) {
            // Fall back to MODIFY path, below.
            break;
          } else {
            return false;
          }
        case ENOENT:
          if (has(mode, WriteMode::CREATE_PARENT) && path.size() > 0 &&
              tryMkdir(path.parent(), WriteMode::CREATE | WriteMode::MODIFY |
                                      WriteMode::CREATE_PARENT, true)) {
            // Retry, but make sure we don't try to create the parent again.
            return tryReplaceNode(path, mode - WriteMode::CREATE_PARENT, kj::mv(tryCreate));
          }
          // fallthrough
        default:
          KJ_FAIL_SYSCALL("create(path)", error, path) { return false; }
      } else {
        // Success.
        return true;
      }
    }

    // Either we don't have CREATE mode or the target already exists. We need to perform a
    // replacement instead.

    KJ_IF_MAYBE(tempPath, createNamedTemporary(path, mode, kj::mv(tryCreate))) {
      const auto indir = kj::mv(*tempPath);
      if (tryCommitReplacement(filename, fd.get_path(), indir, mode)) {
        return true;
      } else {
        const auto p = fs_join(fd.get_path(), tempPath->cStr());
        KJ_SYSCALL_HANDLE_ERRORS(unlink(p.c_str())) {
          case ENOENT:
            // meh
            break;
          default:
            KJ_FAIL_SYSCALL("unlink(tempPath, 0)", error, *tempPath);
        }
        return false;
      }
    } else {
      // threw, but exceptions are disabled
      return false;
    }
  }

  Maybe<AutoCloseFd> tryOpenFileInternal(PathPtr path, WriteMode mode, bool append) const {
    uint flags = O_RDWR | MAYBE_O_CLOEXEC;
    mode_t acl = 0666;
    if (has(mode, WriteMode::CREATE)) {
      flags |= O_CREAT;
    }
    if (!has(mode, WriteMode::MODIFY)) {
      if (!has(mode, WriteMode::CREATE)) {
        // Neither CREATE nor MODIFY -- impossible to satisfy preconditions.
        return nullptr;
      }
      flags |= O_EXCL;
    }
    if (append) {
      flags |= O_APPEND;
    }
    if (has(mode, WriteMode::EXECUTABLE)) {
      acl = 0777;
    }
    if (has(mode, WriteMode::PRIVATE)) {
      acl &= 0700;
    }

    auto filename = path.toString();
    const auto p = fs_join(fd.get_path(), filename.cStr());

    int newFd;
    KJ_SYSCALL_HANDLE_ERRORS(newFd = open(p.c_str(), flags, acl)) {
      case ENOENT:
        if (has(mode, WriteMode::CREATE)) {
          // Either:
          // - The file is a broken symlink.
          // - A parent directory didn't exist.
          if (has(mode, WriteMode::CREATE_PARENT) && path.size() > 0 &&
              tryMkdir(path.parent(), WriteMode::CREATE | WriteMode::MODIFY |
                                      WriteMode::CREATE_PARENT, true)) {
            // Retry, but make sure we don't try to create the parent again.
            return tryOpenFileInternal(path, mode - WriteMode::CREATE_PARENT, append);
          }

          // Check for broken link.
          const auto p = fs_join(fd.get_path(), filename.cStr());
          if (!has(mode, WriteMode::MODIFY) &&
              access(p.c_str(), F_OK) >= 0) {
            // Yep. We treat this as already-exists, which means in CREATE-only mode this is a
            // simple failure.
            return nullptr;
          }

          KJ_FAIL_REQUIRE("parent is not a directory", path) { return nullptr; }
        } else {
          // MODIFY-only mode. ENOENT = doesn't exist = return null.
          return nullptr;
        }
      case ENOTDIR:
        if (!has(mode, WriteMode::CREATE)) {
          // MODIFY-only mode. ENOTDIR = parent not a directory = doesn't exist = return null.
          return nullptr;
        }
        goto failed;
      case EEXIST:
        if (!has(mode, WriteMode::MODIFY)) {
          // CREATE-only mode. EEXIST = already exists = return null.
          return nullptr;
        }
        goto failed;
      default:
      failed:
        KJ_FAIL_SYSCALL("open(path, O_RDWR | ...)", error, p) { return nullptr; }
    }

    kj::AutoCloseFd result(newFd, p);

    return kj::mv(result);
  }

  // jfs: ugh this one is weird
  // Maybe I can promote fromDirFd from an int to an AutoCloseFd
  // Maybe I can `getFdPath` everywhere I need to upstream and can just pass that in, excising the troublesome AutoCloseFd?
  // bool tryCommitReplacement(StringPtr toPath, const AutoCloseFd& fromDirFd, StringPtr fromPath, WriteMode mode,
  //                           int* errorReason = nullptr) const {

  bool tryCommitReplacement(StringPtr toPath, std::string fromRoot, StringPtr fromPath, WriteMode mode,
                            int* errorReason = nullptr) const {
    std::string from = fs_join(fromRoot, fromPath.cStr());
    std::string to = fs_join(fd.get_path(), toPath.cStr());
    if (has(mode, WriteMode::CREATE) && has(mode, WriteMode::MODIFY)) {
      // Always clobber. Try it.
      KJ_SYSCALL_HANDLE_ERRORS(rename(from.c_str(), to.c_str())) {
        case EISDIR:
        case ENOTDIR:
        case ENOTEMPTY:
        case EEXIST:
          // Failed because target exists and due to the various weird quirks of rename(), it
          // can't remove it for us. On Linux we can try an exchange instead. On others we have
          // to move the target out of the way.
          break;
        default:
          if (errorReason == nullptr) {
            KJ_FAIL_SYSCALL("rename(fromPath, toPath)", error, fromPath, toPath) { return false; }
          } else {
            *errorReason = error;
            return false;
          }
      } else {
        return true;
      }
    }

    // We're unable to do what we wanted atomically. :(

    if (has(mode, WriteMode::CREATE) && has(mode, WriteMode::MODIFY)) {
      // We failed to atomically delete the target previously. So now we need to do two calls in
      // rapid succession to move the old file away then move the new one into place.

      // Find out what kind of file exists at the target path.
      struct stat stats;
      KJ_SYSCALL(fstatat(fd, toPath.cStr(), &stats, AT_SYMLINK_NOFOLLOW)) { return false; }

      // Create a temporary location to move the existing object to. Note that rename() allows a
      // non-directory to replace a non-directory, and allows a directory to replace an empty
      // directory. So we have to create the right type.
      Path toPathParsed = Path::parse(toPath);
      String away;
      KJ_IF_MAYBE(awayPath, createNamedTemporary(toPathParsed, WriteMode::CREATE,
          [&](StringPtr candidatePath) {
        const auto mkdir_path = fs_join(fd.get_path(), candidatePath.cStr());
        if (S_ISDIR(stats.st_mode)) {
          return mkdir(mkdir_path.c_str(), 0700);
        } else {
          return mknod(mkdir_path.c_str(), S_IFREG | 0600, dev_t());
        }
      })) {
        away = kj::mv(*awayPath);
      } else {
        // Already threw.
        return false;
      }

      // OK, now move the target object to replace the thing we just created.
      const auto old_to = fs_join(fd.get_path(), away.cStr());
      KJ_SYSCALL(rename(to.c_str(), old_to.c_str())) {
        // Something went wrong. Remove the thing we just created.
        if (S_ISDIR(stats.st_mode))
        {
          rmdir(old_to.c_str());
        }
        else
        {
          unlink(old_to.c_str());
        }
        return false;
      }

      // Now move the source object to the target location.
      KJ_SYSCALL_HANDLE_ERRORS(rename(from.c_str(), to.c_str())) {
        default:
          // Try to put things back where they were. If this fails, though, then we have little
          // choice but to leave things broken.
          KJ_SYSCALL_HANDLE_ERRORS(rename(old_to.c_str(), to.c_str())) {
            default: break;
          }

          if (errorReason == nullptr) {
            KJ_FAIL_SYSCALL("rename(fromPath, toPath)", error, fromPath, toPath) {
              return false;
            }
          } else {
            *errorReason = error;
            return false;
          }
      }

      // OK, success. Delete the old content.
      rmrf(fd, away);
      return true;
    } else {
      // Only one of CREATE or MODIFY is specified, so we need to verify non-atomically that the
      // corresponding precondition (must-not-exist or must-exist, respectively) is held.
      if (has(mode, WriteMode::CREATE)) {
        struct stat stats;
        KJ_SYSCALL_HANDLE_ERRORS(fstatat(fd.get(), toPath.cStr(), &stats, AT_SYMLINK_NOFOLLOW)) {
          case ENOENT:
          case ENOTDIR:
            break;  // doesn't exist; continue
          default:
            KJ_FAIL_SYSCALL("fstatat(fd, toPath)", error, toPath) { return false; }
        } else {
          return false;  // already exists; fail
        }
      } else if (has(mode, WriteMode::MODIFY)) {
        struct stat stats;
        KJ_SYSCALL_HANDLE_ERRORS(fstatat(fd.get(), toPath.cStr(), &stats, AT_SYMLINK_NOFOLLOW)) {
          case ENOENT:
          case ENOTDIR:
            return false;  // doesn't exist; fail
          default:
            KJ_FAIL_SYSCALL("fstatat(fd, toPath)", error, toPath) { return false; }
        } else {
          // already exists; continue
        }
      } else {
        // Neither CREATE nor MODIFY.
        return false;
      }

      // Start over in create-and-modify mode.
      return tryCommitReplacement(toPath, fromRoot, fromPath,
                                  WriteMode::CREATE | WriteMode::MODIFY,
                                  errorReason);
    }
  }

  template <typename T>
  class ReplacerImpl final: public Directory::Replacer<T> {
  public:
    ReplacerImpl(Own<const T>&& object, const DiskHandle& handle,
                 String&& tempPath, String&& path, WriteMode mode)
        : Directory::Replacer<T>(mode),
          object(kj::mv(object)), handle(handle),
          tempPath(kj::mv(tempPath)), path(kj::mv(path)) {}

    ~ReplacerImpl() noexcept(false) {
      if (!committed) {
        rmrf(handle.fd, tempPath);
      }
    }

    const T& get() override {
      return *object;
    }

    bool tryCommit() override {
      KJ_ASSERT(!committed, "already committed") { return false; }
      return committed = handle.tryCommitReplacement(path, handle.fd.get_path(), tempPath,
                                                     Directory::Replacer<T>::mode);
    }

  private:
    Own<const T> object;
    const DiskHandle& handle;
    String tempPath;
    String path;
    bool committed = false;  // true if *successfully* committed (in which case tempPath is gone)
  };

  template <typename T>
  class BrokenReplacer final: public Directory::Replacer<T> {
    // For recovery path when exceptions are disabled.

  public:
    BrokenReplacer(Own<const T> inner)
        : Directory::Replacer<T>(WriteMode::CREATE | WriteMode::MODIFY),
          inner(kj::mv(inner)) {}

    const T& get() override { return *inner; }
    bool tryCommit() override { return false; }

  private:
    Own<const T> inner;
  };

  Maybe<Own<const File>> tryOpenFile(PathPtr path, WriteMode mode) const {
    return tryOpenFileInternal(path, mode, false).map(newDiskFile);
  }

  Own<Directory::Replacer<File>> replaceFile(PathPtr path, WriteMode mode) const {
    mode_t acl = 0666;
    if (has(mode, WriteMode::EXECUTABLE)) {
      acl = 0777;
    }
    if (has(mode, WriteMode::PRIVATE)) {
      acl &= 0700;
    }

    int newFd_;
    KJ_IF_MAYBE(temp, createNamedTemporary(path, mode,
        [&](StringPtr candidatePath) {
      const auto p = fs_join(fd.get_path(), candidatePath.cStr());
      return newFd_ = open(p.c_str(),
                           O_RDWR | O_CREAT | O_EXCL | MAYBE_O_CLOEXEC, acl);
    })) {
      AutoCloseFd newFd(newFd_, temp->cStr());
      return heap<ReplacerImpl<File>>(newDiskFile(kj::mv(newFd)), *this, kj::mv(*temp),
                                      path.toString(), mode);
    } else {
      // threw, but exceptions are disabled
      return heap<BrokenReplacer<File>>(newInMemoryFile(nullClock()));
    }
  }

  Own<const File> createTemporary() const {
    int newFd_;

    KJ_IF_MAYBE(temp, createNamedTemporary(Path("unnamed"), WriteMode::CREATE,
        [&](StringPtr path) {
      const auto p = fs_join(fd.get_path(), path.cStr());
      return newFd_ = open(p.c_str(), O_RDWR | O_CREAT | O_EXCL | MAYBE_O_CLOEXEC, 0600);
    })) {
      AutoCloseFd newFd(newFd_, temp->cStr());
      auto result = newDiskFile(kj::mv(newFd));

      const auto p = fs_join(fd.get_path(), temp->cStr());
      KJ_SYSCALL(p.c_str(), 0) { break; }
      return kj::mv(result);
    } else {
      // threw, but exceptions are disabled
      return newInMemoryFile(nullClock());
    }
  }

  Maybe<Own<AppendableFile>> tryAppendFile(PathPtr path, WriteMode mode) const {
    return tryOpenFileInternal(path, mode, true).map(newDiskAppendableFile);
  }

  Maybe<Own<const Directory>> tryOpenSubdir(PathPtr path, WriteMode mode) const {
    // Must create before open.
    if (has(mode, WriteMode::CREATE)) {
      if (!tryMkdir(path, mode, false)) return nullptr;
    }

    return tryOpenSubdirInternal(path).map(newDiskDirectory);
  }

  Own<Directory::Replacer<Directory>> replaceSubdir(PathPtr path, WriteMode mode) const {
    mode_t acl = has(mode, WriteMode::PRIVATE) ? 0700 : 0777;

    KJ_IF_MAYBE(temp, createNamedTemporary(path, mode,
        [&](StringPtr candidatePath) {
      const auto p = fs_join(fd.get_path(), candidatePath.cStr());
      return mkdir(p.c_str(), acl);
    })) {
      int subdirFd_;
      const auto p = fs_join(fd.get_path(), temp->cStr());
      KJ_SYSCALL_HANDLE_ERRORS(subdirFd_ = open(
            p.c_str(), O_RDONLY | MAYBE_O_CLOEXEC | MAYBE_O_DIRECTORY)) {
        default:
          KJ_FAIL_SYSCALL("open(just-created-temporary)", error);
          return heap<BrokenReplacer<Directory>>(newInMemoryDirectory(nullClock()));
      }

      AutoCloseFd subdirFd(subdirFd_, temp->cStr());
      return heap<ReplacerImpl<Directory>>(
          newDiskDirectory(kj::mv(subdirFd)), *this, kj::mv(*temp), path.toString(), mode);
    } else {
      // threw, but exceptions are disabled
      return heap<BrokenReplacer<Directory>>(newInMemoryDirectory(nullClock()));
    }
  }

  bool trySymlink(PathPtr linkpath, StringPtr content, WriteMode mode) const {
    return tryReplaceNode(linkpath, mode, [&](StringPtr candidatePath) {
      const auto p = fs_join(fd.get_path(), candidatePath.cStr());
      return symlink(content.cStr(), p.c_str());
    });
  }

  bool tryTransfer(PathPtr toPath, WriteMode toMode,
                   const Directory& fromDirectory, PathPtr fromPath,
                   TransferMode mode, const Directory& self) const {
    KJ_REQUIRE(toPath.size() > 0, "can't replace self") { return false; }

    if (mode == TransferMode::LINK) {
      KJ_IF_MAYBE(fromFd, fromDirectory.getFd()) {
        // Other is a disk directory, so we can hopefully do an efficient move/link.
        return tryReplaceNode(toPath, toMode, [&](StringPtr candidatePath) {
          const auto from = fs_join(fromDirectory.getFdPath(), fromPath.toString().cStr());
          const auto to = fs_join(fd.get_path(), candidatePath.cStr());
          return link(from.c_str(), to.c_str());
        });
      };
    } else if (mode == TransferMode::MOVE) {
      KJ_IF_MAYBE(fromFd, fromDirectory.getFd()) {
        KJ_ASSERT(mode == TransferMode::MOVE);

        int error = 0;
        if (tryCommitReplacement(toPath.toString(), fromDirectory.getFdPath(), fromPath.toString(), toMode,
                                 &error)) {
          return true;
        } else switch (error) {
          case 0:
            // Plain old WriteMode precondition failure.
            return false;
          case EXDEV:
            // Can't move between devices. Fall back to default implementation, which does
            // copy/delete.
            break;
          case ENOENT:
            // Either the destination directory doesn't exist or the source path doesn't exist.
            // Unfortunately we don't really know. If CREATE_PARENT was provided, try creating
            // the parent directory. Otherwise, we don't actually need to distinguish between
            // these two errors; just return false.
            if (has(toMode, WriteMode::CREATE) && has(toMode, WriteMode::CREATE_PARENT) &&
                toPath.size() > 0 && tryMkdir(toPath.parent(),
                    WriteMode::CREATE | WriteMode::MODIFY | WriteMode::CREATE_PARENT, true)) {
              // Retry, but make sure we don't try to create the parent again.
              return tryTransfer(toPath, toMode - WriteMode::CREATE_PARENT,
                                 fromDirectory, fromPath, mode, self);
            }
            return false;
          default:
            KJ_FAIL_SYSCALL("rename(fromPath, toPath)", error, fromPath, toPath) {
              return false;
            }
        }
      }
    }

    // OK, we can't do anything efficient using the OS. Fall back to default implementation.
    return self.Directory::tryTransfer(toPath, toMode, fromDirectory, fromPath, mode);
  }

  bool tryRemove(PathPtr path) const {
    return rmrf(fd, path.toString());
  }

protected:
  AutoCloseFd fd;
};

#define FSNODE_METHODS(classname)                                             \
  Maybe<int> getFd() const override { return DiskHandle::getFd(); }           \
                                                                              \
  std::string getFdPath() const override { return DiskHandle::getFdPath(); }  \
                                                                              \
  Own<const FsNode> cloneFsNode() const override {                            \
    return heap<classname>(DiskHandle::clone());                              \
  }                                                                           \
                                                                              \
  Metadata stat() const override { return DiskHandle::stat(); }               \
  void sync() const override { DiskHandle::sync(); }                          \
  void datasync() const override { DiskHandle::datasync(); }

class DiskReadableFile final: public ReadableFile, public DiskHandle {
public:
  DiskReadableFile(AutoCloseFd&& fd): DiskHandle(kj::mv(fd)) {}

  FSNODE_METHODS(DiskReadableFile);

  size_t read(uint64_t offset, ArrayPtr<byte> buffer) const override {
    return DiskHandle::read(offset, buffer);
  }
  Array<const byte> mmap(uint64_t offset, uint64_t size) const override {
    return DiskHandle::mmap(offset, size);
  }
  Array<byte> mmapPrivate(uint64_t offset, uint64_t size) const override {
    return DiskHandle::mmapPrivate(offset, size);
  }
};

class DiskAppendableFile final: public AppendableFile, public DiskHandle, public FdOutputStream {
public:
  DiskAppendableFile(AutoCloseFd&& fd)
      : DiskHandle(kj::mv(fd)),
        FdOutputStream(DiskHandle::fd.get()) {}

  FSNODE_METHODS(DiskAppendableFile);

  void write(const void* buffer, size_t size) override {
    FdOutputStream::write(buffer, size);
  }
  void write(ArrayPtr<const ArrayPtr<const byte>> pieces) override {
    FdOutputStream::write(pieces);
  }
};

class DiskFile final: public File, public DiskHandle {
public:
  DiskFile(AutoCloseFd&& fd): DiskHandle(kj::mv(fd)) {}

  FSNODE_METHODS(DiskFile);

  size_t read(uint64_t offset, ArrayPtr<byte> buffer) const override {
    return DiskHandle::read(offset, buffer);
  }
  Array<const byte> mmap(uint64_t offset, uint64_t size) const override {
    return DiskHandle::mmap(offset, size);
  }
  Array<byte> mmapPrivate(uint64_t offset, uint64_t size) const override {
    return DiskHandle::mmapPrivate(offset, size);
  }

  void write(uint64_t offset, ArrayPtr<const byte> data) const override {
    DiskHandle::write(offset, data);
  }
  void zero(uint64_t offset, uint64_t size) const override {
    DiskHandle::zero(offset, size);
  }
  void truncate(uint64_t size) const override {
    DiskHandle::truncate(size);
  }
  Own<const WritableFileMapping> mmapWritable(uint64_t offset, uint64_t size) const override {
    return DiskHandle::mmapWritable(offset, size);
  }
  size_t copy(uint64_t offset, const ReadableFile& from,
              uint64_t fromOffset, uint64_t size) const override {
    KJ_IF_MAYBE(result, DiskHandle::copy(offset, from, fromOffset, size)) {
      return *result;
    } else {
      return File::copy(offset, from, fromOffset, size);
    }
  }
};

class DiskReadableDirectory final: public ReadableDirectory, public DiskHandle {
public:
  DiskReadableDirectory(AutoCloseFd&& fd): DiskHandle(kj::mv(fd)) {}

  FSNODE_METHODS(DiskReadableDirectory);

  Array<String> listNames() const override { return DiskHandle::listNames(); }
  Array<Entry> listEntries() const override { return DiskHandle::listEntries(); }
  bool exists(PathPtr path) const override { return DiskHandle::exists(path); }
  Maybe<FsNode::Metadata> tryLstat(PathPtr path) const override {
    return DiskHandle::tryLstat(path);
  }
  Maybe<Own<const ReadableFile>> tryOpenFile(PathPtr path) const override {
    return DiskHandle::tryOpenFile(path);
  }
  Maybe<Own<const ReadableDirectory>> tryOpenSubdir(PathPtr path) const override {
    return DiskHandle::tryOpenSubdir(path);
  }
  Maybe<String> tryReadlink(PathPtr path) const override { return DiskHandle::tryReadlink(path); }
};

class DiskDirectory final: public Directory, public DiskHandle {
public:
  DiskDirectory(AutoCloseFd&& fd): DiskHandle(kj::mv(fd)) {}

  FSNODE_METHODS(DiskDirectory);

  Array<String> listNames() const override { return DiskHandle::listNames(); }
  Array<Entry> listEntries() const override { return DiskHandle::listEntries(); }
  bool exists(PathPtr path) const override { return DiskHandle::exists(path); }
  Maybe<FsNode::Metadata> tryLstat(PathPtr path) const override {
    return DiskHandle::tryLstat(path);
  }
  Maybe<Own<const ReadableFile>> tryOpenFile(PathPtr path) const override {
    return DiskHandle::tryOpenFile(path);
  }
  Maybe<Own<const ReadableDirectory>> tryOpenSubdir(PathPtr path) const override {
    return DiskHandle::tryOpenSubdir(path);
  }
  Maybe<String> tryReadlink(PathPtr path) const override { return DiskHandle::tryReadlink(path); }

  Maybe<Own<const File>> tryOpenFile(PathPtr path, WriteMode mode) const override {
    return DiskHandle::tryOpenFile(path, mode);
  }
  Own<Replacer<File>> replaceFile(PathPtr path, WriteMode mode) const override {
    return DiskHandle::replaceFile(path, mode);
  }
  Own<const File> createTemporary() const override {
    return DiskHandle::createTemporary();
  }
  Maybe<Own<AppendableFile>> tryAppendFile(PathPtr path, WriteMode mode) const override {
    return DiskHandle::tryAppendFile(path, mode);
  }
  Maybe<Own<const Directory>> tryOpenSubdir(PathPtr path, WriteMode mode) const override {
    return DiskHandle::tryOpenSubdir(path, mode);
  }
  Own<Replacer<Directory>> replaceSubdir(PathPtr path, WriteMode mode) const override {
    return DiskHandle::replaceSubdir(path, mode);
  }
  bool trySymlink(PathPtr linkpath, StringPtr content, WriteMode mode) const override {
    return DiskHandle::trySymlink(linkpath, content, mode);
  }
  bool tryTransfer(PathPtr toPath, WriteMode toMode,
                   const Directory& fromDirectory, PathPtr fromPath,
                   TransferMode mode) const override {
    return DiskHandle::tryTransfer(toPath, toMode, fromDirectory, fromPath, mode, *this);
  }
  // tryTransferTo() not implemented because we have nothing special we can do.
  bool tryRemove(PathPtr path) const override {
    return DiskHandle::tryRemove(path);
  }
};

class DiskFilesystem final: public Filesystem {
public:
  DiskFilesystem()
      : root(openDir("/")),
        current(openDir(".")),
        currentPath(computeCurrentPath()) {}

  const Directory& getRoot() const override {
    return root;
  }

  const Directory& getCurrent() const override {
    return current;
  }

  PathPtr getCurrentPath() const override {
    return currentPath;
  }

private:
  DiskDirectory root;
  DiskDirectory current;
  Path currentPath;

  static AutoCloseFd openDir(const char* dir) {
    int newFd;
    KJ_SYSCALL(newFd = open(dir, O_RDONLY | MAYBE_O_CLOEXEC | MAYBE_O_DIRECTORY));
    AutoCloseFd result(newFd, dir);
    return result;
  }

  static Path computeCurrentPath() {
    // If env var PWD is set and points to the current directory, use it. This captures the current
    // path according to the user's shell, which may differ from the kernel's idea in the presence
    // of symlinks.
    const char* pwd = getenv("PWD");
    if (pwd != nullptr) {
      Path result = nullptr;
      struct stat pwdStat, dotStat;
      KJ_IF_MAYBE(e, kj::runCatchingExceptions([&]() {
        KJ_ASSERT(pwd[0] == '/') { return; }
        result = Path::parse(pwd + 1);
        KJ_SYSCALL(lstat(result.toString(true).cStr(), &pwdStat), result) { return; }
        KJ_SYSCALL(lstat(".", &dotStat)) { return; }
      })) {
        // failed, give up on PWD
        KJ_LOG(WARNING, "PWD environment variable seems invalid", pwd, *e);
      } else {
        if (pwdStat.st_ino == dotStat.st_ino &&
            pwdStat.st_dev == dotStat.st_dev) {
          return kj::mv(result);
        } else {
          KJ_LOG(WARNING, "PWD environment variable doesn't match current directory", pwd);
        }
      }
    }

    size_t size = 256;
  retry:
    KJ_STACK_ARRAY(char, buf, size, 256, 4096);
    if (getcwd(buf.begin(), size) == nullptr) {
      int error = errno;
      if (error == ENAMETOOLONG) {
        size *= 2;
        goto retry;
      } else {
        KJ_FAIL_SYSCALL("getcwd()", error);
      }
    }

    StringPtr path = buf.begin();

    // On Linux, the path will start with "(unreachable)" if the working directory is not a subdir
    // of the root directory, which is possible via chroot() or mount namespaces.
    KJ_ASSERT(!path.startsWith("(unreachable)"),
        "working directory is not reachable from root", path);
    KJ_ASSERT(path.startsWith("/"), "current directory is not absolute", path);

    return Path::parse(path.slice(1));
  }
};

} // namespace

Own<ReadableFile> newDiskReadableFile(kj::AutoCloseFd fd) {
  return heap<DiskReadableFile>(kj::mv(fd));
}
Own<AppendableFile> newDiskAppendableFile(kj::AutoCloseFd fd) {
  return heap<DiskAppendableFile>(kj::mv(fd));
}
Own<File> newDiskFile(kj::AutoCloseFd fd) {
  return heap<DiskFile>(kj::mv(fd));
}
Own<ReadableDirectory> newDiskReadableDirectory(kj::AutoCloseFd fd) {
  return heap<DiskReadableDirectory>(kj::mv(fd));
}
Own<Directory> newDiskDirectory(kj::AutoCloseFd fd) {
  return heap<DiskDirectory>(kj::mv(fd));
}

Own<Filesystem> newDiskFilesystem() {
  return heap<DiskFilesystem>();
}

} // namespace kj

#endif  // !_WIN32
