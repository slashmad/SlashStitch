//! Atomic publication. The destination is never opened/truncated during rendering.
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tempfile::{Builder, NamedTempFile};

pub fn protect_sources(
    destination: &Path,
    sources: impl IntoIterator<Item = impl AsRef<Path>>,
) -> io::Result<()> {
    let absolute = absolute_destination(destination)?;
    for source in sources {
        let source = source.as_ref();
        let canonical = source
            .canonicalize()
            .or_else(|_| absolute_destination(source))?;
        if canonical == absolute
            || (destination.exists()
                && source.exists()
                && same_file::is_same_file(source, destination)?)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Destination is a source image (including a link or alias)",
            ));
        }
    }
    // Replacing a symlink is ambiguous and can bypass a prior file-picker decision.
    if std::fs::symlink_metadata(destination).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Destination must not be a symbolic link",
        ));
    }
    Ok(())
}

pub fn absolute_destination(path: &Path) -> io::Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "A file name is required"))?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(parent.canonicalize()?.join(name))
}

pub fn temporary_output(path: &Path) -> io::Result<NamedTempFile> {
    let destination = absolute_destination(path)?;
    Builder::new()
        .prefix(".slashstitch-")
        .suffix(".part")
        .tempfile_in(destination.parent().unwrap())
}

pub fn publish(mut temp: NamedTempFile, path: &Path, replace: bool) -> io::Result<()> {
    temp.flush()?;
    temp.as_file().sync_all()?;
    if replace {
        temp.persist(path).map_err(|e| e.error)?;
    } else {
        temp.persist_noclobber(path).map_err(|e| e.error)?;
    }
    #[cfg(unix)]
    if let Err(error) = std::fs::File::open(absolute_destination(path)?.parent().unwrap())
        .and_then(|directory| directory.sync_all())
    {
        // Publication already committed. A durability warning must not turn a
        // complete output into a reported failed/cancelled render.
        eprintln!("Output committed; directory durability sync failed: {error}");
    }
    Ok(())
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut temp = temporary_output(path)?;
    temp.write_all(bytes)?;
    publish(temp, path, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protects_aliases_and_does_not_clobber() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.tif");
        let alias = dir.path().join("alias.tif");
        std::fs::write(&source, b"original").unwrap();
        std::fs::hard_link(&source, &alias).unwrap();
        assert!(protect_sources(&alias, [&source]).is_err());
        assert!(protect_sources(&source, [&source]).is_err());
        let mut temp = temporary_output(&alias).unwrap();
        temp.write_all(b"replacement").unwrap();
        assert!(publish(temp, &alias, false).is_err());
        assert_eq!(std::fs::read(&source).unwrap(), b"original");
        write_atomic(&alias, b"new inode").unwrap();
        assert_eq!(std::fs::read(&source).unwrap(), b"original");
        assert_eq!(std::fs::read(&alias).unwrap(), b"new inode");
    }
}
