#pragma once

// DocumentLock: the reader/writer lock that guards one Document while a job runs.
//
// A job worker holds it exclusively for the whole run of a task; front-end code that only reads
// the document (models, property sheets, saving) takes it shared, for the duration of one call,
// and copies what it needs. The lock is not stored inside Document so the data model stays
// copyable; whoever owns the document owns the lock.

#include <mutex>
#include <shared_mutex>

namespace alphashape {

class DocumentLock {
public:
    /// Held by the one writer (a running task).
    std::unique_lock<std::shared_mutex> write() {
        return std::unique_lock<std::shared_mutex>(mutex_);
    }
    /// Held briefly by readers.
    std::shared_lock<std::shared_mutex> read() {
        return std::shared_lock<std::shared_mutex>(mutex_);
    }
    /// Non-blocking probes; the returned locks are empty (owns_lock() == false) on failure.
    std::unique_lock<std::shared_mutex> try_write() {
        return std::unique_lock<std::shared_mutex>(mutex_, std::try_to_lock);
    }
    std::shared_lock<std::shared_mutex> try_read() {
        return std::shared_lock<std::shared_mutex>(mutex_, std::try_to_lock);
    }

private:
    std::shared_mutex mutex_;
};

}  // namespace alphashape
