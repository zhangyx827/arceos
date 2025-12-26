use alloc::{
    boxed::Box, sync::{Arc, Weak}, vec::Vec
};
#[cfg(feature = "times")]
use core::sync::atomic::{AtomicU8, Ordering};
use core::{num::NonZeroUsize, ops::Range, task::Context};

use axalloc::{UsageKind, global_allocator};
use axfs_ng_vfs::{
    FileNode, Location, NodeFlags, NodePermission, NodeType, VfsError, VfsResult, path::Path,
};
use axhal::mem::{PhysAddr, VirtAddr, phys_to_virt, virt_to_phys};
use axio::{Buf, BufMut, SeekFrom};
use axpoll::{IoEvents, Pollable};
use intrusive_collections::{LinkedList, LinkedListAtomicLink, intrusive_adapter};
use lru::LruCache;
use spin::{Mutex, MutexGuard, Once, RwLock};

use super::FsContext;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct FileFlags: u8 {
        const READ = 1;
        const WRITE = 2;
        const EXECUTE = 4;
        const APPEND = 8;
        const PATH = 16;
    }
}

/// Results returned by [`OpenOptions::open`].
pub enum OpenResult {
    File(File),
    Dir(Location),
}

impl OpenResult {
    pub fn into_file(self) -> VfsResult<File> {
        match self {
            Self::File(file) => Ok(file),
            Self::Dir(_) => Err(VfsError::IsADirectory),
        }
    }

    pub fn into_dir(self) -> VfsResult<Location> {
        match self {
            Self::Dir(dir) => Ok(dir),
            Self::File(_) => Err(VfsError::NotADirectory),
        }
    }

    pub fn into_location(self) -> Location {
        match self {
            Self::File(file) => file.location().clone(),
            Self::Dir(dir) => dir,
        }
    }
}

/// Options and flags which can be used to configure how a file is opened.
#[derive(Debug, Clone)]
pub struct OpenOptions {
    // generic
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    directory: bool,
    no_follow: bool,
    direct: bool,
    user: Option<(u32, u32)>,
    path: bool,
    node_type: NodeType,
    // system-specific
    mode: u32,
}

impl OpenOptions {
    /// Creates a blank new set of options ready for configuration.
    pub fn new() -> Self {
        Self {
            // generic
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            directory: false,
            no_follow: false,
            direct: false,
            user: None,
            path: false,
            node_type: NodeType::RegularFile,
            // system-specific
            mode: 0o666,
        }
    }

    /// Sets the option for read access.
    pub fn read(&mut self, read: bool) -> &mut Self {
        self.read = read;
        self
    }

    /// Sets the option for write access.
    pub fn write(&mut self, write: bool) -> &mut Self {
        self.write = write;
        self
    }

    /// Sets the option for the append mode.
    pub fn append(&mut self, append: bool) -> &mut Self {
        self.append = append;
        self
    }

    /// Sets the option for truncating a previous file.
    pub fn truncate(&mut self, truncate: bool) -> &mut Self {
        self.truncate = truncate;
        self
    }

    /// Sets the option to create a new file, or open it if it already exists.
    pub fn create(&mut self, create: bool) -> &mut Self {
        self.create = create;
        self
    }

    /// Sets the option to create a new file, failing if it already exists.
    pub fn create_new(&mut self, create_new: bool) -> &mut Self {
        self.create_new = create_new;
        self
    }

    /// Sets the option to open directory instead.
    pub fn directory(&mut self, directory: bool) -> &mut Self {
        self.directory = directory;
        self
    }

    /// Sets the option to not follow symlinks.
    pub fn no_follow(&mut self, no_follow: bool) -> &mut Self {
        self.no_follow = no_follow;
        self
    }

    /// Sets the option to open the file with direct I/O.\
    pub fn direct(&mut self, direct: bool) -> &mut Self {
        self.direct = direct;
        self
    }

    /// Sets the user and group id to open the file with.
    pub fn user(&mut self, uid: u32, gid: u32) -> &mut Self {
        self.user = Some((uid, gid));
        self
    }

    /// Sets the option for path only access.
    pub fn path(&mut self, path: bool) -> &mut Self {
        self.path = path;
        self
    }

    /// Sets the node type for the file.
    ///
    /// This will only be used if the file is created.
    pub fn node_type(&mut self, node_type: NodeType) -> &mut Self {
        self.node_type = node_type;
        self
    }

    /// Sets the mode bits that a new file will be created with.
    pub fn mode(&mut self, mode: u32) -> &mut Self {
        self.mode = mode;
        self
    }

    fn _open(&self, loc: Location) -> VfsResult<OpenResult> {
        let flags = self.to_flags()?;

        if self.directory {
            if flags.contains(FileFlags::WRITE) {
                return Err(VfsError::IsADirectory);
            }
            loc.check_is_dir()?;
        }
        if self.truncate {
            loc.entry().as_file()?.set_len(0)?;
        }

        Ok(if loc.is_dir() {
            OpenResult::Dir(loc)
        } else {
            // TODO(mivik): is this correct?
            let non_cacheable_type = matches!(
                loc.metadata()?.node_type,
                NodeType::CharacterDevice | NodeType::Fifo | NodeType::Socket
            );

            let direct = non_cacheable_type
                || self.path
                || self.direct
                || loc.flags().contains(NodeFlags::NON_CACHEABLE);
            let backend = if !direct || loc.flags().contains(NodeFlags::ALWAYS_CACHE) {
                FileBackend::new_cached(loc)
            } else {
                FileBackend::new_direct(loc)
            };
            OpenResult::File(File::new(backend, flags))
        })
    }

    pub fn open_loc(&self, loc: Location) -> VfsResult<OpenResult> {
        if !self.is_valid() {
            return Err(VfsError::InvalidInput);
        }
        self._open(loc)
    }

    pub fn open(&self, context: &FsContext, path: impl AsRef<Path>) -> VfsResult<OpenResult> {
        if !self.is_valid() {
            return Err(VfsError::InvalidInput);
        }

        let loc = match context.resolve_parent(path.as_ref()) {
            Ok((parent, name)) => {
                let mut loc = parent.open_file(
                    &name,
                    &axfs_ng_vfs::OpenOptions {
                        create: self.create,
                        create_new: self.create_new,
                        node_type: self.node_type,
                        permission: NodePermission::from_bits_truncate(self.mode as _),
                        user: self.user,
                    },
                )?;
                if !self.no_follow {
                    loc = context
                        .with_current_dir(parent)?
                        .try_resolve_symlink(loc, &mut 0)?;
                }
                loc
            }
            Err(VfsError::InvalidInput) => {
                // root directory
                context.root_dir().clone()
            }
            Err(err) => return Err(err),
        };
        self._open(loc)
    }

    pub(crate) fn to_flags(&self) -> VfsResult<FileFlags> {
        Ok(match (self.read, self.write, self.append) {
            (true, false, false) => FileFlags::READ,
            (false, true, false) => FileFlags::WRITE,
            (true, true, false) => FileFlags::READ | FileFlags::WRITE,
            (false, _, true) => FileFlags::WRITE | FileFlags::APPEND,
            (true, _, true) => FileFlags::READ | FileFlags::WRITE | FileFlags::APPEND,
            (false, false, false) => return Err(VfsError::InvalidInput),
        } | if self.path {
            FileFlags::PATH
        } else {
            FileFlags::empty()
        })
    }

    pub(crate) fn is_valid(&self) -> bool {
        if !self.read && !self.write && !self.append {
            return true;
        }
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return false;
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self::new()
    }
}

const PAGE_SIZE_4K: usize = 4096;
const PAGE_SIZE_2M: usize = 2 * 1024 * 1024;
const THP_NR_4K_PAGES: usize = PAGE_SIZE_2M / PAGE_SIZE_4K;

#[derive(Debug, Clone, Copy)]
pub enum PageOperation {
    /// Unmap the page from page table
    Unmap,
    /// Demote huge page mapping from PMD to PTE
    Demote,
}

#[derive(Debug)]
pub struct PageCache {
    addr: VirtAddr,
    dirty: bool,
    size: usize,
}
impl PageCache {
    fn new(size: usize) -> VfsResult<Self> {
        let addr = global_allocator()
            .alloc_pages(size / PAGE_SIZE_4K, size, UsageKind::PageCache)
            .inspect_err(|err| {
                warn!("Failed to allocate page cache: {:?}", err);
            })?;
        Ok(Self {
            addr: addr.into(),
            dirty: false,
            size,
        })
    }

    /// # Safety
    /// - `addr` must be aligned to `PAGE_SIZE_4K`, and `size` must be a
    ///   multiple of `PAGE_SIZE_4K`.
    /// - The range `[addr, addr + size)` must refer to pages allocated from
    ///   `global_allocator().alloc_pages(...)`.
    /// - This `PageCache` must be the unique owner responsible for
    ///   deallocating that range in `Drop` with `num_pages = size /
    ///   PAGE_SIZE_4K` (i.e. no other object will deallocate the same pages,
    ///   such as the original huge-page owner).
    unsafe fn from_addr_unchecked(addr: VirtAddr, size: usize) -> Self {
        Self {
            addr,
            size,
            dirty: false,
        }
    }

    pub fn paddr(&self) -> PhysAddr {
        virt_to_phys(self.addr)
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn data(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.addr.as_mut_ptr(), self.size) }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PageCache {
    fn drop(&mut self) {
        if self.dirty {
            warn!("dirty page dropped without flushing");
        }
        global_allocator().dealloc_pages(
            self.addr.as_usize(),
            self.size / PAGE_SIZE_4K,
            UsageKind::PageCache,
        );
    }
}

struct EvictListener {
    listener: Box<dyn Fn(u32, &PageCache, PageOperation) + Send + Sync>,
    link: LinkedListAtomicLink,
}

intrusive_adapter!(EvictListenerAdapter = Box<EvictListener>: EvictListener { link: LinkedListAtomicLink });

struct CachedFileShared {
    page_cache: Mutex<LruCache<u32, PageCache>>,
    evict_listeners: Mutex<LinkedList<EvictListenerAdapter>>,
    in_memory: bool,
}

impl CachedFileShared {
    pub fn new(in_memory: bool) -> Self {
        Self {
            // page_cache: Mutex::new(LruCache::unbounded()),
            page_cache: Mutex::new(LruCache::new(NonZeroUsize::new(256).unwrap())),
            evict_listeners: Mutex::new(LinkedList::default()),
            in_memory,
        }
    }

    pub fn new_unbounded(in_memory: bool) -> Self {
        Self {
            page_cache: Mutex::new(LruCache::unbounded()),
            evict_listeners: Mutex::new(LinkedList::default()),
            in_memory,
        }
    }
}

pub struct CachedFile {
    inner: Location,
    shared: Arc<CachedFileShared>,
    in_memory: bool,
    /// Only one thread can append to the file at a time, while multiple writers
    /// are permitted.
    append_lock: RwLock<()>,
}

impl Clone for CachedFile {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            shared: self.shared.clone(),
            in_memory: self.in_memory,
            append_lock: RwLock::new(()),
        }
    }
}

enum FileUserData {
    Weak(Weak<CachedFileShared>),
    Strong(Arc<CachedFileShared>),
}

impl FileUserData {
    pub fn get(&self) -> Option<Arc<CachedFileShared>> {
        match self {
            FileUserData::Weak(weak) => weak.upgrade(),
            FileUserData::Strong(strong) => Some(strong.clone()),
        }
    }
}

impl CachedFile {
    pub fn get_or_create(location: Location) -> Self {
        let in_memory = location.filesystem().name() == "tmpfs";

        let mut guard = location.user_data();
        let shared = if let Some(shared) = guard.get::<FileUserData>().and_then(|it| it.get()) {
            shared
        } else {
            let (shared, user_data) = if in_memory {
                let shared = Arc::new(CachedFileShared::new_unbounded(true));
                (shared.clone(), FileUserData::Strong(shared))
            } else {
                let shared = Arc::new(CachedFileShared::new(false));
                let user_data = FileUserData::Weak(Arc::downgrade(&shared));
                (shared, user_data)
            };
            guard.insert(user_data);
            register_page_cache(&shared);
            shared
        };
        drop(guard);

        Self {
            inner: location,
            shared,
            in_memory,
            append_lock: RwLock::new(()),
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }

    pub fn in_memory(&self) -> bool {
        self.in_memory
    }

    pub fn add_evict_listener<F>(&self, listener: F) -> usize
    where
        F: Fn(u32, &PageCache, PageOperation) + Send + Sync + 'static,
    {
        let pointer = Box::new(EvictListener {
            listener: Box::new(listener),
            link: LinkedListAtomicLink::new(),
        });
        let handle = pointer.as_ref() as *const EvictListener as usize;
        self.shared.evict_listeners.lock().push_back(pointer);
        handle
    }

    pub unsafe fn remove_evict_listener(&self, handle: usize) {
        let mut guard = self.shared.evict_listeners.lock();
        let mut cursor = unsafe { guard.cursor_mut_from_ptr(handle as *const EvictListener) };
        cursor.remove();
    }

    fn evict_cache(&self, file: &FileNode, pn: u32, page: &mut PageCache) -> VfsResult<()> {
        for listener in self.shared.evict_listeners.lock().iter() {
            (listener.listener)(pn, page, PageOperation::Unmap);
        }
        if page.dirty {
            let page_start = pn as u64 * PAGE_SIZE_4K as u64;
            let file_len = file.len()?;
            if page_start < file_len {
                let len = (file_len - page_start).min(page.size() as u64) as usize;
                file.write_at(&page.data()[..len], page_start)?;
            }
            page.dirty = false;
        }
        Ok(())
    }

    pub fn locate_chunk(&self, pn: u32) -> (Option<u32>, usize) {
        let mut page_num = 0;
        let mut diff = 1;
        while page_num <= pn {
            let (opt_num, page_size) = self.with_page(page_num, |opt_page| {
                if let Some(ref page) = opt_page {
                    if page.size() == PAGE_SIZE_2M
                        && page_num + (PAGE_SIZE_2M / PAGE_SIZE_4K) as u32 > pn
                    {
                        return (Some(page_num), PAGE_SIZE_2M);
                    }
                    if page.size() == PAGE_SIZE_4K && page_num == pn {
                        return (Some(page_num), PAGE_SIZE_4K);
                    }
                }
                diff = opt_page.map(|p| p.size() / PAGE_SIZE_4K).unwrap_or(1) as u32;
                return (None, 0);
            });
            if opt_num.is_some() {
                return (opt_num, page_size);
            } else {
                page_num += diff;
            }
        }
        return (None, PAGE_SIZE_4K);
    }

    fn page_or_insert<'a>(
        &self,
        file: &FileNode,
        cache: &'a mut LruCache<u32, PageCache>,
        pn: u32,
        size: usize,
    ) -> VfsResult<(&'a mut PageCache, Option<(u32, PageCache)>)> {
        // TODO: Matching the result of `get_mut` confuses compiler. See
        // https://users.rust-lang.org/t/return-do-not-release-mutable-borrow/55757.
        if cache.contains(&pn) {
            return Ok((cache.get_mut(&pn).unwrap(), None));
        }
        let mut evicted = None;
        if cache.len() == cache.cap().get() {
            // Cache is full, remove the least recently used page
            if let Some((pn, mut page)) = cache.pop_lru() {
                self.evict_cache(file, pn, &mut page)?;
                evicted = Some((pn, page));
            }
        }

        // Page not in cache, read it
        let mut page = PageCache::new(size)?;
        if self.in_memory {
            page.data().fill(0);
        } else {
            file.read_at(page.data(), pn as u64 * PAGE_SIZE_4K as u64)?;
        }
        cache.put(pn, page);
        Ok((cache.get_mut(&pn).unwrap(), evicted))
    }

    pub fn with_page<R>(&self, pn: u32, f: impl FnOnce(Option<&mut PageCache>) -> R) -> R {
        f(self.shared.page_cache.lock().get_mut(&pn))
    }

    pub fn with_page_or_insert<R>(
        &self,
        pn: u32,
        size: usize,
        f: impl FnOnce(&mut PageCache, Option<(u32, PageCache)>) -> VfsResult<R>,
    ) -> VfsResult<R> {
        let mut guard = self.shared.page_cache.lock();
        let (page, evicted) =
            self.page_or_insert(self.inner.entry().as_file()?, &mut guard, pn, size)?;
        f(page, evicted)
    }

    fn with_pages<T>(
        &self,
        range: Range<u64>,
        page_initial: impl FnOnce(&FileNode) -> VfsResult<T>,
        mut page_each: impl FnMut(T, &mut PageCache, Range<usize>) -> VfsResult<T>,
    ) -> VfsResult<T> {
        let file = self.inner.entry().as_file()?;
        let mut initial = page_initial(file)?;

        let mut size = PAGE_SIZE_4K as u64;
        // Skip the chunks before the range
        let (mut pn, mut chunk_start) = self.locate_offset(range.start);

        while chunk_start < range.end {
            let mut guard = self.shared.page_cache.lock();
            let page = self.page_or_insert(file, &mut guard, pn, PAGE_SIZE_4K)?.0;
            let size = page.size() as u64;
            let chunk_end = chunk_start + size;

            let read_start = core::cmp::max(chunk_start, range.start);
            let read_end = core::cmp::min(chunk_end, range.end);

            if read_start < read_end {
                let page_offset = (read_start - chunk_start) as usize;
                let len = (read_end - read_start) as usize;
                initial = page_each(initial, page, page_offset..page_offset + len)?;
            }

            chunk_start = chunk_end;
            pn += size as u32 / PAGE_SIZE_4K as u32;
        }

        Ok(initial)
    }

    pub fn read_at(&self, dst: &mut impl BufMut, offset: u64) -> VfsResult<usize> {
        let len = self.inner.len()?;
        let end = (offset + dst.remaining_mut() as u64).min(len);
        if end <= offset {
            return Ok(0);
        }
        self.with_pages(
            offset..end,
            |_| Ok(0),
            |read, page, range| {
                let len = range.end - range.start;
                dst.write(&page.data()[range.start..range.end])?;
                Ok(read + len)
            },
        )
    }

    fn write_at_locked(&self, buf: &mut impl Buf, offset: u64) -> VfsResult<usize> {
        let end = offset + buf.remaining() as u64;
        self.with_pages(
            offset..end,
            |file| {
                if end > file.len()? {
                    file.set_len(end)?;
                }
                Ok(0)
            },
            |written, page, range| {
                let len = range.end - range.start;
                buf.read(&mut page.data()[range.start..range.end])?;
                if !self.in_memory {
                    page.dirty = true;
                }
                Ok(written + len)
            },
        )
    }

    pub fn write_at(&self, buf: &mut impl Buf, offset: u64) -> VfsResult<usize> {
        let _guard = self.append_lock.read();
        self.write_at_locked(buf, offset)
    }

    pub fn append(&self, buf: &mut impl Buf) -> VfsResult<(usize, u64)> {
        let _guard = self.append_lock.write();
        let file = self.inner.entry().as_file()?;
        let len = file.len()?;
        self.write_at_locked(buf, len)
            .map(|written| (written, len + written as u64))
    }

    pub fn set_len(&self, len: u64) -> VfsResult<()> {
        let file = self.inner.entry().as_file()?;
        let old_len = file.len()?;
        file.set_len(len)?;
        let min_len = core::cmp::min(len, old_len);
        // Skip the chunks before the min_len
        let (pn, chunk_start) = self.locate_offset(min_len);

        if old_len < len {
            // pn is the old page num
            let mut guard = self.shared.page_cache.lock();
            if let Some(page) = guard.get_mut(&pn) {
                let page_start = chunk_start;
                let old_page_offset = (old_len - page_start) as usize;
                let new_page_offset = (len - page_start).min(page.size() as u64) as usize;
                page.data()[old_page_offset..new_page_offset].fill(0);
            }
        } else {
            // For truncating, we need to remove all pages that are beyond the
            // new length
            // TODO(mivik): can this be more efficient?
            let mut guard = self.shared.page_cache.lock();
            let keys = guard
                .iter()
                .map(|(k, _)| *k)
                .filter(|it| *it > pn)
                .collect::<Vec<_>>();

            for pn in keys {
                if let Some(mut page) = guard.pop(&pn) {
                    if !self.in_memory {
                        // Don't write back pages since they're discarded
                        page.dirty = false;
                        self.evict_cache(file, pn, &mut page)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn len(&self) -> VfsResult<u64> {
        self.inner.entry().as_file()?.len()
    }

    pub fn sync(&self, data_only: bool) -> VfsResult<()> {
        if self.in_memory {
            return Ok(());
        }
        let file = self.inner.entry().as_file()?;
        let mut guard = self.shared.page_cache.lock();
        while let Some((pn, mut page)) = guard.pop_lru() {
            self.evict_cache(file, pn, &mut page)?;
        }
        file.sync(data_only)?;
        Ok(())
    }

    pub fn location(&self) -> &Location {
        &self.inner
    }

    pub fn retract_page_tables(
        &self,
        start_pn: u32,
        guard: &mut MutexGuard<'_, LruCache<u32, PageCache>>,
    ) {
        let listeners = self.shared.evict_listeners.lock();
        for listener in listeners.iter() {
            for i in 0..THP_NR_4K_PAGES as u32 {
                let pn = start_pn + i;
                if let Some(page) = guard.get_mut(&pn) {
                    (listener.listener)(pn, page, PageOperation::Unmap);
                }
            }
        }
    }

    pub fn replace_with_huge_page(&self, start_pn: u32) -> VfsResult<PhysAddr> {
        let mut guard = self.shared.page_cache.lock();
        self.retract_page_tables(start_pn, &mut guard);
        let page = PageCache::new(PAGE_SIZE_2M)?;
        let new_pa = page.paddr();

        for i in 0..THP_NR_4K_PAGES as u32 {
            let pn = start_pn + i;
            let dst_off = i as usize * PAGE_SIZE_4K;
            if let Some(page) = guard.get_mut(&pn) {
                let src_pa = page.paddr();
                let src = phys_to_virt(src_pa);
                let dst = phys_to_virt(new_pa + dst_off);
                unsafe {
                    core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), PAGE_SIZE_4K);
                }
            } else {
                // This is a hole
            }
        }

        for i in 0..THP_NR_4K_PAGES as u32 {
            if let Some(mut page) = guard.pop(&(start_pn + i)) {
                if !self.in_memory {
                    // Don't write back pages since they're discarded
                    page.dirty = false;
                }
            }
        }
        // Since some caches have been evicted
        // we don't have to evict cache here
        guard.put(start_pn, page);
        Ok(new_pa)
    }

    pub fn split_huge_page(&self, pn: u32) -> VfsResult<usize> {
        let (base_pn_opt, _) = self.locate_chunk(pn);
        let base_pn = base_pn_opt.unwrap();
        let mut guard = self.shared.page_cache.lock();
        let page = guard.get(&base_pn).unwrap();
        let start_va = phys_to_virt(page.paddr());
        let file = self.inner.entry().as_file()?;
        let huge_page = guard.pop(&base_pn).unwrap();

        for listener in self.shared.evict_listeners.lock().iter() {
            (listener.listener)(base_pn, &huge_page, PageOperation::Demote);
        }
        
        core::mem::forget(huge_page);
        
        for i in 0..THP_NR_4K_PAGES as u32 {
            if guard.len() == guard.cap().get() {
                // Cache is full, remove the least recently used page
                if let Some((pn, mut page)) = guard.pop_lru() {
                    // NOTE: Currently we only collapse read-only mappings, so
                    // pages are never marked dirty. Therefore `evict_cache()`
                    // won't perform writeback and is not expected to return an
                    // error here. 
                    if let Err(err) = self.evict_cache(file, pn, &mut page) {
                        guard.put(pn, page);
                        return Err(err);
                    }
                }
            }

            // SAFETY: `addr` points to the i-th 4KiB subpage of the 2MiB page we
            // just removed from the cache; we also `forget(huge_page)` above so
            // the 2MiB owner will not deallocate these pages. Each subpage is
            // wrapped exactly once and will be deallocated by `PageCache::Drop`
            // with `num_pages = 1`.
            let addr = start_va + i as usize * PAGE_SIZE_4K;
            let new_page = unsafe { PageCache::from_addr_unchecked(addr, PAGE_SIZE_4K) };
            guard.put(base_pn + i, new_page);
        }
        Ok(0)
    }
    

    /// Find the cache page that covers `offset` (bytes), returning (page_index,
    /// page_start_offset).
    pub fn locate_offset(&self, offset: u64) -> (u32, u64) {
        let mut pn = 0;
        let mut chunk_start = 0;
        let mut size = 0;
        while chunk_start < offset {
            size = self.with_page(pn, |opt_page| {
                opt_page.map(|p| p.size()).unwrap_or(PAGE_SIZE_4K)
            }) as u64;

            let chunk_end = chunk_start + size;
            if chunk_end > offset {
                break;
            }
            chunk_start = chunk_end;
            pn += size as u32 / PAGE_SIZE_4K as u32;
        }
        (pn, chunk_start)
    }
}

static PAGE_CACHE_REGISTRY: Once<Mutex<Vec<Weak<CachedFileShared>>>> = Once::new();

fn page_cache_registry() -> &'static Mutex<Vec<Weak<CachedFileShared>>> {
    PAGE_CACHE_REGISTRY.call_once(|| Mutex::new(Vec::new()))
}

fn register_page_cache(shared: &Arc<CachedFileShared>) {
    let mut caches = page_cache_registry().lock();
    caches.push(Arc::downgrade(shared));
}

/// Best-effort global page cache reclaim.
///
/// This only evicts clean pages from non-tmpfs caches that currently have no
/// registered eviction listeners (i.e. are not mmap'd anywhere), so it is safe
/// to call even while holding an address space lock.
///
/// Returns the number of 4KiB pages freed.
pub fn shrink_page_cache(target_pages: usize) -> usize {
    if target_pages == 0 {
        return 0;
    }

    let caches_snapshot = page_cache_registry().lock().clone();
    let mut freed_pages = 0usize;

    for weak in caches_snapshot {
        if freed_pages >= target_pages {
            break;
        }
        let Some(shared) = weak.upgrade() else { continue; };
        if shared.in_memory {
            continue;
        }
        if !shared.evict_listeners.lock().is_empty() {
            continue;
        }

        let mut dirty_skips = 0usize;
        loop {
            if freed_pages >= target_pages {
                break;
            }

            let popped = {
                let mut guard = shared.page_cache.lock();
                if guard.is_empty() {
                    None
                } else {
                    match guard.pop_lru() {
                        None => None,
                        Some((pn, page)) if page.dirty => {
                            guard.put(pn, page);
                            dirty_skips += 1;
                            if dirty_skips >= guard.len() {
                                // No evictable pages in this cache.
                                None
                            } else {
                                Some(None)
                            }
                        }
                        Some((_pn, page)) => {
                            dirty_skips = 0;
                            Some(Some(page))
                        }
                    }
                }
            };

            match popped {
                None => break,
                Some(None) => continue, // skipped a dirty page
                Some(Some(page)) => {
                    let pages = page.size() / PAGE_SIZE_4K;
                    freed_pages += pages;
                    // drop(page) here frees the underlying memory
                }
            }
        }
    }

    // Best-effort cleanup of dead entries.
    page_cache_registry().lock().retain(|w| w.upgrade().is_some());

    freed_pages
}

impl Drop for CachedFile {
    fn drop(&mut self) {
        if Arc::strong_count(&self.shared) > 1 {
            // If there are other references to this cached file, we don't
            // need to drop it.
            return;
        }
        if let Err(err) = self.sync(false) {
            warn!("Failed to sync file on drop: {err:?}");
        }
    }
}

/// Low-level interface for file operations.
#[derive(Clone)]
pub enum FileBackend {
    Cached(CachedFile),
    Direct(Location),
}

impl FileBackend {
    pub(crate) fn new_direct(location: Location) -> Self {
        Self::Direct(location)
    }

    pub(crate) fn new_cached(location: Location) -> Self {
        Self::Cached(CachedFile::get_or_create(location))
    }

    pub fn read_at(&self, dst: &mut impl BufMut, mut offset: u64) -> VfsResult<usize> {
        match self {
            Self::Cached(cached) => cached.read_at(dst, offset),
            Self::Direct(loc) => dst.fill(|buf| {
                loc.entry().as_file()?.read_at(buf, offset).inspect(|read| {
                    offset += *read as u64;
                })
            }),
        }
    }

    pub fn write_at(&self, src: &mut impl Buf, mut offset: u64) -> VfsResult<usize> {
        match self {
            Self::Cached(cached) => cached.write_at(src, offset),
            Self::Direct(loc) => src.consume(|buf| {
                loc.entry()
                    .as_file()?
                    .write_at(buf, offset)
                    .inspect(|written| {
                        offset += *written as u64;
                    })
            }),
        }
    }

    pub fn append(&self, src: &mut impl Buf) -> VfsResult<(usize, u64)> {
        match self {
            Self::Cached(cached) => cached.append(src),
            Self::Direct(loc) => {
                let mut buffer = Box::<[u8]>::new_uninit_slice(src.remaining());
                src.read(unsafe { buffer.assume_init_mut() })?;
                loc.entry()
                    .as_file()?
                    .append(unsafe { buffer.assume_init_ref() })
            }
        }
    }

    pub fn location(&self) -> &Location {
        match self {
            Self::Cached(cached) => cached.location(),
            Self::Direct(loc) => loc,
        }
    }

    pub fn sync(&self, data_only: bool) -> VfsResult<()> {
        match self {
            Self::Cached(cached) => cached.sync(data_only),
            Self::Direct(loc) => loc.entry().as_file()?.sync(data_only),
        }
    }

    pub fn set_len(&self, len: u64) -> VfsResult<()> {
        match self {
            Self::Cached(cached) => cached.set_len(len),
            Self::Direct(loc) => loc.entry().as_file()?.set_len(len),
        }
    }
}

/// Provides `std::fs::File`-like interface.
pub struct File {
    inner: FileBackend,
    flags: FileFlags,
    position: Option<Mutex<u64>>,
    #[cfg(feature = "times")]
    access_flags: AtomicU8,
}

impl File {
    pub fn new(inner: FileBackend, flags: FileFlags) -> Self {
        let position = if inner.location().flags().contains(NodeFlags::STREAM) {
            None
        } else {
            Some(Mutex::new(if flags.contains(FileFlags::APPEND) {
                inner.location().len().unwrap_or_default()
            } else {
                0
            }))
        };
        Self {
            inner,
            flags,
            position,
            #[cfg(feature = "times")]
            access_flags: AtomicU8::new(0),
        }
    }

    pub fn open(context: &FsContext, path: impl AsRef<Path>) -> VfsResult<Self> {
        OpenOptions::new()
            .read(true)
            .open(context, path.as_ref())
            .and_then(OpenResult::into_file)
    }

    pub fn create(context: &FsContext, path: impl AsRef<Path>) -> VfsResult<Self> {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(context, path.as_ref())
            .and_then(OpenResult::into_file)
    }

    pub fn access(&self, flags: FileFlags) -> VfsResult<&FileBackend> {
        if self.flags.contains(flags) && !self.is_path() {
            Ok(&self.inner)
        } else {
            Err(VfsError::BadFileDescriptor)
        }
    }

    pub fn is_path(&self) -> bool {
        self.flags.contains(FileFlags::PATH)
    }

    pub fn flags(&self) -> FileFlags {
        self.flags
    }

    pub fn backend(&self) -> VfsResult<&FileBackend> {
        self.access(FileFlags::empty())?;
        Ok(&self.inner)
    }

    pub fn location(&self) -> &Location {
        self.inner.location()
    }

    /// Reads a number of bytes starting from a given offset.
    pub fn read_at(&self, dst: &mut impl BufMut, offset: u64) -> VfsResult<usize> {
        self.access(FileFlags::READ)?.read_at(dst, offset)
    }

    /// Writes a number of bytes starting from a given offset.
    pub fn write_at(&self, src: &mut impl Buf, offset: u64) -> VfsResult<usize> {
        self.access(FileFlags::WRITE)?.write_at(src, offset)
    }

    /// Attempts to sync OS-internal file content and metadata to disk.
    ///
    /// If `data_only` is `true`, only the file data is synced, not the
    /// metadata.
    pub fn sync(&self, data_only: bool) -> VfsResult<()> {
        self.access(FileFlags::empty())?;
        self.inner.sync(data_only)
    }

    pub fn read(&self, dst: &mut impl BufMut) -> axio::Result<usize> {
        #[cfg(feature = "times")]
        {
            self.access_flags.fetch_or(1, Ordering::AcqRel);
        }
        if let Some(pos) = self.position.as_ref() {
            let mut pos = pos.lock();
            self.read_at(dst, *pos).inspect(|n| {
                *pos += *n as u64;
            })
        } else {
            self.read_at(dst, 0)
        }
    }

    pub fn write(&self, src: &mut impl Buf) -> axio::Result<usize> {
        #[cfg(feature = "times")]
        {
            self.access_flags.fetch_or(3, Ordering::AcqRel);
        }
        if let Some(pos) = self.position.as_ref() {
            let mut pos = pos.lock();
            if let Ok(f) = self.access(FileFlags::APPEND) {
                f.append(src).map(|(written, new_size)| {
                    *pos = new_size;
                    written
                })
            } else {
                self.write_at(src, *pos).inspect(|n| {
                    *pos += *n as u64;
                })
            }
        } else {
            self.write_at(src, 0)
        }
    }

    pub fn flush(&self) -> axio::Result {
        self.access(FileFlags::empty())?;
        Ok(())
    }
}

impl<'a> axio::Read for &'a File {
    fn read(&mut self, mut buf: &mut [u8]) -> axio::Result<usize> {
        (*self).read(&mut buf)
    }
}

impl<'a> axio::Write for &'a File {
    fn write(&mut self, mut buf: &[u8]) -> axio::Result<usize> {
        (*self).write(&mut buf)
    }

    fn flush(&mut self) -> axio::Result {
        (*self).flush()
    }
}

impl<'a> axio::Seek for &'a File {
    fn seek(&mut self, pos: SeekFrom) -> axio::Result<u64> {
        self.access(FileFlags::empty())?;

        if let Some(guard) = self.position.as_ref() {
            let mut guard = guard.lock();
            let new_pos = match pos {
                SeekFrom::Start(pos) => pos,
                SeekFrom::End(off) => {
                    let size = self.access(FileFlags::empty())?.location().len()?;
                    size.checked_add_signed(off).ok_or(VfsError::InvalidInput)?
                }
                SeekFrom::Current(off) => guard
                    .checked_add_signed(off)
                    .ok_or(VfsError::InvalidInput)?,
            };
            *guard = new_pos;
            Ok(new_pos)
        } else {
            Ok(0)
        }
    }
}

impl Pollable for File {
    fn poll(&self) -> IoEvents {
        self.inner.location().poll()
    }

    fn register(&self, context: &mut Context<'_>, events: IoEvents) {
        self.inner.location().register(context, events)
    }
}

#[cfg(feature = "times")]
impl Drop for File {
    fn drop(&mut self) {
        let flags = self.access_flags.load(Ordering::Acquire);
        if flags != 0 {
            let mut update = axfs_ng_vfs::MetadataUpdate::default();
            if flags & 1 != 0 {
                update.atime = Some(axhal::time::wall_time());
            }
            if flags & 2 != 0 {
                update.mtime = Some(axhal::time::wall_time());
            }
            if let Err(err) = self.inner.location().update_metadata(update) {
                warn!("Failed to update file times on drop: {err:?}");
            }
        }
    }
}
