use env_logger::Builder;
use log::{LevelFilter, error, info};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::task;
use tokio::time::Instant;

/// Index type that maps file signature to paths.
/// The Hash is wrapped into RwLock for multi-threaded write access to the Hash.
/// Arc is used to keep the value on heap with a reference counter allowing sharing the pointer
/// to RwLock between threads and not dropping if from memory ahread of time.
type FileIndexMap = Arc<RwLock<HashMap<Vec<u8>, Vec<String>>>>;

/// Main File Index that contains the mapping of the file hashed to file paths.
/// All non-unique files will be stored as the same hash.
pub struct FileIndex {
    index: FileIndexMap,
}

impl FileIndex {
    pub fn new() -> FileIndex {
        FileIndex {
            index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Recursively add a directory to the index
    pub fn index_dir(&self, dir: String) {
        let files = list_files_recursively(Path::new(dir.as_str()));

        for file in files {
            let path = file.to_string_lossy().into_owned();
            let index = self.index.clone();
            task::spawn_blocking(move || Self::index_path(index, path.clone()));
        }
    }

    /// Add a new file to the index.
    pub fn add_file(&self, file: String) {
        Self::index_path(self.index.clone(), file)
    }

    /// Clear the index.
    pub fn clear(&self) {
        self.index.write().unwrap().clear();
    }

    /// Spawnable task for indexing files.
    fn index_path(index: FileIndexMap, path: String) {
        let file_signature = match FileSignature::new(path.clone()) {
            Ok(file_signature) => file_signature,
            Err(err) => {
                error!("Couldn't index file '{path}' due to error: {err:?}");
                return;
            }
        };
        let mut index = index.write().unwrap();
        if index.contains_key(&file_signature.signature) {
            index.get_mut(&file_signature.signature).unwrap().push(path);
        } else {
            index.insert(file_signature.signature.clone(), vec![path.clone()]);
        }
    }

    /// Returns a list of duplicate files
    pub fn duplicates(&self) -> Vec<(Vec<u8>, Vec<String>)> {
        let res = self
            .index
            .read()
            .unwrap()
            .iter()
            .filter(|(_, files)| files.len() > 1)
            .map(|(hash, files)| (hash.clone(), files.clone()))
            .collect();
        res
    }
}

#[derive(Debug, Clone)]
pub struct FileSignature {
    path: String,
    file_size: u64,
    chunks: Vec<FileChunk>,
    signature: Vec<u8>,
}

impl FileSignature {
    pub fn new(path: String) -> Result<FileSignature, FileIndexError> {
        let size = file_size(path.clone())?;

        let (n_chunks, chunk_size) = Self::chunks_for_size(size);
        let mut chunks = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            match Self::chunk_offset(size.clone(), n_chunks.clone(), chunk_size.clone(), i) {
                None => {}
                Some(offset) => {
                    let chunk_signature =
                        FileChunk::new(path.clone(), offset, chunk_size as usize)?;
                    chunks.push(chunk_signature);
                }
            }
        }

        let chunk_hashes: Vec<u8> = chunks
            .iter()
            .map(|chunk| chunk.hash.clone())
            .flatten()
            .collect();

        Ok(FileSignature {
            path,
            file_size: size,
            chunks,
            signature: hash_chunk(chunk_hashes),
        })
    }

    /// Find offset of a file chunk
    ///
    /// ### Parameters:
    /// * file_size - total file size
    /// * n_chunks - total number of chunks in a file
    /// * chunk_size - size of chunks
    /// * id of the chunk for which to get the offset.
    pub fn chunk_offset(
        file_size: u64,
        n_chunks: usize,
        chunk_size: u64,
        chunk_id: usize,
    ) -> Option<u64> {
        if n_chunks == 0 || chunk_id >= n_chunks {
            return None;
        }
        if file_size == 0 {
            return Some(0);
        }

        // Never let the requested chunk be larger than the file.
        let effective_chunk = chunk_size.min(file_size);
        let max_offset = file_size - effective_chunk;

        // If only one chunk or no room to move, start at 0.
        if n_chunks == 1 || max_offset == 0 {
            return Some(0);
        }

        // Ensure the last chunk is anchored at the tail exactly.
        if chunk_id + 1 == n_chunks {
            return Some(max_offset);
        }

        // Even spacing for intermediate chunks (floor division).
        Some((chunk_id as u64) * max_offset / (n_chunks as u64 - 1))
    }

    /// Determines the chunking strategy depending the on the file size.
    /// For example for small files it will create one big chunk the size of the file.
    /// The bigger the file is the more chunks it will add and the bigger the chunks will be.
    fn chunks_for_size(size: u64) -> (usize, u64) {
        const MB: u64 = 1024 * 1024;
        const GB: u64 = 1024 * MB;

        match size {
            0 => (0, 0),
            x if x <= 1 * MB => (1, x),        // hash whole file
            x if x <= 10 * MB => (2, 1 * MB),  // head + tail
            x if x <= 100 * MB => (3, 1 * MB), // head/mid/tail
            x if x <= 1 * GB => (5, 1 * MB),
            x if x <= 8 * GB => (7, 2 * MB),
            _ => (9, 4 * MB),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileChunk {
    path: String,
    offset: u64,
    len: usize,
    hash: Vec<u8>,
}

impl FileChunk {
    pub fn new(path: String, offset: u64, len: usize) -> Result<FileChunk, FileIndexError> {
        let data = read_chunk(path.clone(), offset, len)?;
        let hash = hash_chunk(data);

        Ok(FileChunk {
            path,
            offset,
            len,
            hash,
        })
    }
}

#[derive(Debug, Clone)]
pub enum FileIndexError {
    FileDoesNotExist(String),
    FileReadError(String, String),
}

fn file_size(path: String) -> Result<u64, FileIndexError> {
    let file_path = Path::new(&path);

    if !file_path.exists() {
        return Err(FileIndexError::FileDoesNotExist(path));
    }

    let metadata = fs::metadata(file_path)
        .map_err(|err| FileIndexError::FileReadError(path, err.to_string()))?;
    let file_size = metadata.len();

    Ok(file_size)
}

fn read_chunk(path: String, offset: u64, len: usize) -> Result<Vec<u8>, FileIndexError> {
    let mut buf = vec![0u8; len];
    let file = File::open(path.clone())
        .map_err(|err| FileIndexError::FileReadError(path.clone(), err.to_string()))?;

    file.read_exact_at(&mut buf, offset)
        .map_err(|err| FileIndexError::FileReadError(path, err.to_string()))?;

    Ok(buf)
}

fn hash_chunk(chunk: Vec<u8>) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(chunk);
    let hash_result = hasher.finalize();
    hash_result.to_vec()
}

fn list_files_recursively(path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            } else if path.is_dir() {
                // Recurse into subdirectory
                files.extend(list_files_recursively(&path));
            }
        }
    }

    files
}

#[tokio::main(flavor = "multi_thread", worker_threads = 100)]
async fn main() {
    let mut builder = Builder::new();
    builder.filter_level(LevelFilter::Info);
    builder.init();

    let index = FileIndex::new();
    let args: Vec<String> = std::env::args().collect();

    let path = if args.len() > 1 {
        args.get(1).unwrap()
    } else {
        info!("No arguments provided. Indexing current location");
        "./"
    };

    let start = Instant::now();
    index.index_dir(Path::new(path).to_str().unwrap().to_string());

    info!("Finished indexing in {}ms", start.elapsed().as_millis());

    let dups = index.duplicates();

    info!("Duplicate hash counts");
    for (hash, files) in dups.iter() {
        info!("{hash:x?} - {}", files.len());
    }

    info!("Duplicate hash reports");
    for (hash, files) in dups.iter() {
        info!("-----{hash:X?}-----");
        for file in files.iter() {
            info!("{file}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn uniq_temp_dir() -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("fileindex_test_{}", nanos));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_temp_file(dir: &Path, name: &str, data: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(data).unwrap();
        path
    }

    #[test]
    fn test_chunks_for_size_boundaries() {
        const MB: u64 = 1024 * 1024;
        const GB: u64 = 1024 * MB;

        assert_eq!(FileSignature::chunks_for_size(0), (0, 0));
        assert_eq!(FileSignature::chunks_for_size(1 * MB), (1, 1 * MB));
        assert_eq!(FileSignature::chunks_for_size(1 * MB + 1), (2, 1 * MB));
        assert_eq!(FileSignature::chunks_for_size(10 * MB), (2, 1 * MB));
        assert_eq!(FileSignature::chunks_for_size(10 * MB + 1), (3, 1 * MB));
        assert_eq!(FileSignature::chunks_for_size(100 * MB), (3, 1 * MB));
        assert_eq!(FileSignature::chunks_for_size(100 * MB + 1), (5, 1 * MB));
        assert_eq!(FileSignature::chunks_for_size(1 * GB), (5, 1 * MB));
        assert_eq!(FileSignature::chunks_for_size(1 * GB + 1), (7, 2 * MB));
        assert_eq!(FileSignature::chunks_for_size(8 * GB), (7, 2 * MB));
        assert_eq!(FileSignature::chunks_for_size(8 * GB + 1), (9, 4 * MB));
    }

    #[test]
    fn test_chunk_offset_edge_cases() {
        // No chunks or bad id
        assert_eq!(FileSignature::chunk_offset(100, 0, 50, 0), None);
        assert_eq!(FileSignature::chunk_offset(100, 3, 50, 3), None);

        // Zero-sized file -> offset 0
        assert_eq!(FileSignature::chunk_offset(0, 1, 64, 0), Some(0));

        // Single chunk always starts at 0
        assert_eq!(FileSignature::chunk_offset(1234, 1, 256, 0), Some(0));

        // Chunk larger than file -> max_offset = 0 -> 0
        assert_eq!(FileSignature::chunk_offset(100, 3, 1000, 0), Some(0));
        assert_eq!(FileSignature::chunk_offset(100, 3, 1000, 1), Some(0));
        assert_eq!(FileSignature::chunk_offset(100, 3, 1000, 2), Some(0));
    }

    #[test]
    fn test_chunk_offset_spacing_and_tail_anchor() {
        assert_eq!(FileSignature::chunk_offset(1000, 3, 100, 0), Some(0));
        assert_eq!(FileSignature::chunk_offset(1000, 3, 100, 1), Some(450));
        assert_eq!(FileSignature::chunk_offset(1000, 3, 100, 2), Some(900));

        assert_eq!(FileSignature::chunk_offset(1000, 5, 100, 0), Some(0));
        assert_eq!(FileSignature::chunk_offset(1000, 5, 100, 1), Some(225));
        assert_eq!(FileSignature::chunk_offset(1000, 5, 100, 2), Some(450));
        assert_eq!(FileSignature::chunk_offset(1000, 5, 100, 3), Some(675));
        assert_eq!(FileSignature::chunk_offset(1000, 5, 100, 4), Some(900));
    }

    #[test]
    fn test_read_and_hash_chunk() {
        let dir = uniq_temp_dir();
        let data = b"abcdefghijklmnopqrstuvwxyz";
        let path = write_temp_file(&dir, "test_read_and_hash_chunk.txt", data);

        // Read "cde"
        let got = read_chunk(path.to_string_lossy().into(), 2, 3).unwrap();
        assert_eq!(got, b"cde");

        // Hash "cde" using our function and direct Sha256 for parity
        let ours = hash_chunk(b"cde".to_vec());

        let mut h = Sha256::new();
        h.update(b"cde");
        let direct = h.finalize().to_vec();

        assert_eq!(ours, direct);

        // Cleanup
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_file_signature_single_chunk_double_hash() {
        let dir = uniq_temp_dir();
        let data = b"small file content";
        let path = write_temp_file(&dir, "test_file_signature_single_chunk_double_hash.txt", data);
        let sig = FileSignature::new(path.to_string_lossy().into()).unwrap();

        // For small files (<=1MB), n_chunks=1 and chunk_size=size
        assert_eq!(sig.file_size as usize, data.len());
        assert_eq!(sig.chunks.len(), 1);

        // Expected signature = SHA256( concat( SHA256(file_data) ) )
        let inner = {
            let mut h = Sha256::new();
            h.update(data);
            h.finalize().to_vec()
        };
        let expected = {
            let mut h = Sha256::new();
            h.update(&inner);
            h.finalize().to_vec()
        };

        assert_eq!(sig.signature, expected);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_list_files_recursively_finds_nested() {
        let dir = uniq_temp_dir();
        write_temp_file(&dir, "a.txt", b"a");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        write_temp_file(&sub, "b.txt", b"b");
        let sub2 = sub.join("sub2");
        fs::create_dir_all(&sub2).unwrap();
        write_temp_file(&sub2, "c.txt", b"c");

        let mut files = list_files_recursively(&dir);
        files.sort();

        assert_eq!(files.len(), 3);
        assert!(files.iter().any(|p| p.ends_with("a.txt")));
        assert!(files.iter().any(|p| p.ends_with("b.txt")));
        assert!(files.iter().any(|p| p.ends_with("c.txt")));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_file_index_duplicates() {
        let dir = uniq_temp_dir();
        let p1 = write_temp_file(&dir, "same1.bin", b"same bytes");
        let p2 = write_temp_file(&dir, "same2.bin", b"same bytes");
        let p3 = write_temp_file(&dir, "diff.bin", b"different");

        let idx = FileIndex::new();
        idx.add_file(p1.to_string_lossy().into());
        idx.add_file(p2.to_string_lossy().into());
        idx.add_file(p3.to_string_lossy().into());

        let mut dups = idx.duplicates();
        assert_eq!(dups.len(), 1);

        // The single duplicate entry should have 2 paths.
        let (_hash, mut files) = dups.pop().unwrap();
        files.sort();

        let p1s = p1.to_string_lossy().into_owned();
        let p2s = p2.to_string_lossy().into_owned();

        assert_eq!(files, vec![p1s, p2s]);

        fs::remove_dir_all(dir).ok();
    }
}
