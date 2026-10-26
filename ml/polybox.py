"""Read public Polybox ZIPs with HTTP ranges, without downloading build artifacts."""

import base64
from collections import OrderedDict
import io
import shutil
import ssl
import tempfile
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import certifi


DEFAULT_SHARE = "https://polybox.ethz.ch/index.php/s/N5f6FHcoFRiCeM3"
TLS_CONTEXT = ssl.create_default_context()
# Extra trusted certificates for macOS Python
TLS_CONTEXT.load_verify_locations(certifi.where())


def _request(url, headers, method="GET"):
    return urlopen(Request(url, headers=headers, method=method), timeout=60, context=TLS_CONTEXT)


def archives(share):
    """Discover ZIPs recursively; names do not determine the application."""
    parsed = urlsplit(share)
    token = parsed.path.rstrip("/").split("/")[-1]
    if parsed.scheme != "https" or "/s/" not in parsed.path or not token:
        raise ValueError(f"expected an HTTPS public Polybox share URL, got {share!r}")
    root = f"{parsed.scheme}://{parsed.netloc}/public.php/webdav/"
    # Share token as WebDAV username; empty password
    auth = base64.b64encode(f"{token}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    pending = [root]
    visited = set()
    while pending:
        folder = pending.pop()
        if folder in visited:
            continue
        visited.add(folder)
        # Immediate children; subfolders queued below
        with _request(folder, {**headers, "Depth": "1"}, "PROPFIND") as response:
            listing = ET.fromstring(response.read())
        for item in listing.findall("{DAV:}response"):
            href = item.findtext("{DAV:}href")
            if not href:
                continue
            url = urljoin(root, href)
            # Credentials restricted to this share's WebDAV root
            if not url.startswith(root):
                continue
            for propstat in item.findall("{DAV:}propstat"):
                if " 200 " not in propstat.findtext("{DAV:}status", ""):
                    continue
                prop = propstat.find("{DAV:}prop")
                if prop.find("{DAV:}resourcetype/{DAV:}collection") is not None:
                    pending.append(url)
                elif unquote(urlsplit(url).path).lower().endswith(".zip"):
                    yield url, int(prop.findtext("{DAV:}getcontentlength")), headers


class RemoteZip(io.RawIOBase):
    """Seekable HTTP file for zipfile; cache at most 4 MiB of 64 KiB blocks."""

    BLOCK = 65536

    def __init__(self, url, size, headers):
        super().__init__()
        self.url, self.size, self.headers = url, size, headers
        self.position = 0
        self.blocks = OrderedDict()
        self.download = None

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset + {io.SEEK_SET: 0, io.SEEK_CUR: self.position,
                             io.SEEK_END: self.size}[whence]
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _block(self, start):
        if start in self.blocks:
            # Recently used blocks kept in cache
            self.blocks.move_to_end(start)
            return self.blocks[start]
        end = min(start + self.BLOCK, self.size) - 1
        headers = {**self.headers, "Range": f"bytes={start}-{end}",
                   "Accept-Encoding": "identity"}
        with _request(self.url, headers) as response:
            if response.status == 200:
                # Full-download fallback for servers without ranges
                self.download = tempfile.TemporaryFile()
                shutil.copyfileobj(response, self.download)
                self.download.seek(start)
                return self.download.read(end - start + 1)
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise OSError(f"invalid HTTP range response for {self.url}")
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError(f"incomplete HTTP range response for {self.url}")
        self.blocks[start] = data
        if len(self.blocks) > 64:
            self.blocks.popitem(last=False)
        return data

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while size:
            if self.download is not None:
                self.download.seek(self.position)
                data = self.download.read(size)
            else:
                # Requested bytes from aligned 64 KiB blocks
                start = self.position // self.BLOCK * self.BLOCK
                block = self._block(start)
                offset = self.position - start
                data = block[offset:offset + size]
            if not data:
                raise OSError(f"unexpected end of remote ZIP: {self.url}")
            chunks.append(data)
            self.position += len(data)
            size -= len(data)
        return b"".join(chunks)

    def close(self):
        if self.download is not None:
            self.download.close()
        self.blocks.clear()
        super().close()
