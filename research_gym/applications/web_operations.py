import requests
import time
import json
import os
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin
import re


class WebOperations:
    """Web browsing operations client, communicates with web_server via HTTP"""
    
    def __init__(self, server_host: str = 'localhost', server_port: int = 8124, 
                 proxy_url: str = None, cache_dir: str = './web_cache', 
                 cache_duration_days: int = 1, max_output_length: int = 30000):
        # Server configuration
        self.server_host = server_host
        self.server_port = server_port
        self.proxy_url = proxy_url if proxy_url else None
        
        if proxy_url:
            self.base_url = proxy_url
            self.target_host = f"{server_host}:{server_port}"
        else:
            self.base_url = f"http://{server_host}:{server_port}"
            self.target_host = f"{server_host}:{server_port}"
        
        # Cache configuration
        self.cache_dir = cache_dir
        self.cache_duration_days = cache_duration_days
        self.cache_duration_seconds = cache_duration_days * 24 * 3600
        
        # Output truncation configuration (mainly for page display operations)
        self.max_output_length = max_output_length  # Maximum output characters

        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.current_url: Optional[str] = None
        self.current_line: int = 1
        self.window: int = 100

        # Web-specific state
        self.cached_text: List[str] = []  # Cached page text lines
        self.cached_links: List[Dict[str, str]] = []  # Cached links
        self.search_results: List[Dict[str, Any]] = []  # Search results
        self.keyword: str = ""  # Search keyword
        self.current_search_index: int = 0  # Current search result index
        
        self.last_health_check = None
    
    def _get_cache_filename(self, url: str) -> str:
        """Generate cache filename based on URL"""
        # Convert URL to safe filename
        import hashlib
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"web_cache_{url_hash}.json")
    
    def _load_from_cache(self, url: str) -> Optional[Dict[str, Any]]:
        """Load webpage data from cache"""
        cache_file = self._get_cache_filename(url)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            cache_time = cache_data.get('cache_time', 0)
            if time.time() - cache_time > self.cache_duration_seconds:
                # Cache expired, delete file
                os.remove(cache_file)
                return None

            return cache_data
        except Exception:
            # Cache file corrupted, delete it
            try:
                os.remove(cache_file)
            except Exception:
                pass
            return None
    
    def _save_to_cache(self, url: str, cached_text: List[str], cached_links: List[Dict]) -> None:
        """Save webpage data to cache"""
        cache_file = self._get_cache_filename(url)
        
        cache_data = {
            'url': url,
            'cached_text': cached_text,
            'cached_links': cached_links,
            'cache_time': time.time()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache for {url}: {e}")
    
    def _make_request(self, method: str, endpoint: str, data: dict = None, timeout: int = 60) -> dict:
        """Send HTTP request to server"""
        if self.last_health_check is None or time.time() - self.last_health_check > 3600:
            if endpoint != '/health':
                if not self._check_health():    
                    raise Exception("Health check failed, please check the web server is running")
        
        url = urljoin(self.base_url, endpoint)
        
        if self.proxy_url:  
            headers = {
                'Content-Type': 'application/json',
                'X-TARGET-HOST': self.target_host
            }
        else:
            headers = { 
                'Content-Type': 'application/json'
            }
            
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=data, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)
            else:
                return None

            response.raise_for_status()
            try:
                return response.json() if response.content else {}
            except ValueError:
                # Return empty dictionary for non-JSON responses to avoid crashes
                return {}
        except requests.RequestException as e:
            print(f"HTTP request failed: {e}")
            return None
    
    def _check_health(self) -> bool:
        """Check server health status"""
        try:
            response = self._make_request('GET', '/health', timeout=5)
            if not response or not response.get('status'):
                raise Exception(f"Failed to check health: {response.get('error', 'Unknown error')}")
        except Exception as e:
            if self.proxy_url:
                print(f"Error checking health: {e}, please check the web server proxy: {self.proxy_url} is running")
            else:
                print(f"Error checking health: {e}, please check the web server is running")
            return False
        self.last_health_check = time.time()
        return True
    
    def _output_error(self, error_msg: str) -> Dict[str, Any]:
        """Output error message, mimicking OpenHands _output_error"""
        return {"message": error_msg, "success": False}
    
    def _check_current_page(self) -> bool:
        """Check if current page is valid"""
        return self.current_url is not None and len(self.cached_text) > 0
    
    def _clamp(self, value: int, min_value: int, max_value: int) -> int:
        """Limit numerical range, mimicking OpenHands _clamp"""
        return max(min_value, min(value, max_value))
    
    def _print_window(self, targeted_line: int, window: int, 
                     ignore_window: bool = False) -> Dict[str, Any]:
        """Display page window, mimicking OpenHands _print_window"""
        if not self._check_current_page():
            return self._output_error('No page loaded. Use the `web_browse` first.')
        
        total_lines = len(self.cached_text)
        if total_lines == 0:
            return self._output_error('Page appears to be empty.')
        
        # Adjust current line
        self.current_line = self._clamp(targeted_line, 1, total_lines)
        half_window = max(1, window // 2)
        
        if ignore_window:
            # For scroll_down/scroll_up
            start = max(1, self.current_line)
            end = min(total_lines, self.current_line + window - 1)
        else:
            # Normal window mode
            start = max(1, self.current_line - half_window)
            end = min(total_lines, self.current_line + half_window)
        
        # Adjust range to ensure enough lines are displayed
        if start == 1:
            end = min(total_lines, start + window - 1)
        if end == total_lines:
            start = max(1, end - window + 1)
        
        # Build output
        output_lines = []
        
        # Page header information
        header = f'[Page: {self.current_url} ({total_lines} lines total)]'
        output_lines.append(header)
        
        # Prompt above
        if start > 1:
            output_lines.append(f'({start - 1} more lines above)')
        else:
            output_lines.append('(this is the beginning of the page)')
        
        # Page content
        page_content = []
        for i in range(start, end + 1):
            line_content = self.cached_text[i - 1] if i <= len(self.cached_text) else ''
            page_content.append(f'{i}|{line_content}')
        
        # Prompt below
        if end < total_lines:
            output_lines.append(f'({total_lines - end} more lines below)')
            # Add scroll prompt
            output_lines.append('[Use `web_page_scroll_down` to view the next 100 lines of the page!]')
        else:
            output_lines.append('(this is the end of the page)')
        
        # Build complete output and truncate
        full_output = "\n".join(output_lines[:-1] + page_content + [output_lines[-1]])
        full_output = self._truncate_string(full_output)
        
        return {
            "success": True,
            "url": self.current_url,
            "current_line": self.current_line,
            "total_lines": total_lines,
            "start_line": start,
            "end_line": end,
            "header": header,
            "content": page_content,
            "output": full_output
        }
    
    def _check_unsupported_url(self, url: str) -> Optional[str]:
        """Check if the URL points to an unsupported file type (such as PDF, document, etc.)"""
        import urllib.parse
        
        # Parse URL
        parsed_url = urllib.parse.urlparse(url.lower())
        path = parsed_url.path
        
        # Check file extension
        unsupported_extensions = {
            '.pdf': 'PDF document',
            '.doc': 'Word document', 
            '.docx': 'Word document',
            '.xls': 'Excel spreadsheet',
            '.xlsx': 'Excel spreadsheet', 
            '.ppt': 'PowerPoint presentation',
            '.pptx': 'PowerPoint presentation',
            '.zip': 'ZIP archive',
            '.rar': 'RAR archive',
            '.tar': 'TAR archive',
            '.gz': 'GZIP archive',
            '.exe': 'executable file',
            '.dmg': 'disk image',
            '.iso': 'disk image',
            '.mp4': 'video file',
            '.avi': 'video file',
            '.mov': 'video file',
            '.mp3': 'audio file',
            '.wav': 'audio file',
            '.jpg': 'image file',
            '.jpeg': 'image file', 
            '.png': 'image file',
            '.gif': 'image file',
            '.svg': 'image file'
        }
        
        for ext, file_type in unsupported_extensions.items():
            if path.endswith(ext):
                return file_type
        
        # Check specific domain and path patterns
        if 'arxiv.org/pdf/' in url:
            return 'arXiv PDF document'
        elif '/pdf/' in path and (path.endswith('.pdf') or not '.' in path.split('/')[-1]):
            return 'PDF document'
        elif 'github.com' in parsed_url.netloc and '/releases/download/' in path:
            return 'downloadable file'
        
        return None
    
    def _clear_page_state(self):
        """Clear the current page state"""
        self.current_url = None
        self.current_line = 1
        self.cached_text = []
        self.cached_links = []
        self.search_results = []
        self.keyword = ""
        self.current_search_index = 0

    def goto(self, url: str, line_number: int = 1) -> Dict[str, Any]:
        """Navigate to a webpage based on URL and display its content.
        
        The environment will cache the webpage content for another action to use until perform next web_browse action.
        
        Args:
            url[str]: The URL to navigate to.
            line_number[int]: The line number to start viewing from. The environment will perform line_number to line_number + 100 lines of content. Default is 1.
            
        Returns:
            Dict[str, Any]: Dictionary containing page content and status information.
        """
        try:
            # Check if it's an unsupported file type
            unsupported_type = self._check_unsupported_url(url)
            if unsupported_type:
                # Clear current page state to avoid showing content from previous page
                self._clear_page_state()
                
                error_msg = f"""Cannot browse {url} directly through web browser.

This appears to be a {unsupported_type} which cannot be scraped as web content.

If you want to download the file, please use the `run_command` to download the file instead of using browse operations."""
                
                return self._output_error(error_msg)
            
            # First check cache
            cache_data = self._load_from_cache(url)
            
            if cache_data:
                # Use cached data
                self.cached_text = cache_data['cached_text']
                self.cached_links = cache_data['cached_links']
                self.current_url = url
                # Clear search state
                self.search_results = []
                self.keyword = ""
                self.current_search_index = 0
                print(f"[Loading page from cache: {url}]")
            else:
                # Cache miss, load from server
                response = self._make_request('POST', '/api/pages', {
                    'url': url,
                    'headless': True
                })
                
                if not response or not response.get('success'):
                    # Clear state when loading fails
                    self._clear_page_state()
                    error_msg = response.get('error', 'Unknown error') if response else 'No response from server'
                    return self._output_error(f'Error loading page {url}: {error_msg}')
                
                # Update local state
                self.cached_text = response['cached_text']
                self.cached_links = response['cached_links']
                self.current_url = url
                # Clear search state
                self.search_results = []
                self.keyword = ""
                self.current_search_index = 0
                
                # Save to cache
                self._save_to_cache(url, self.cached_text, self.cached_links)
                
                # Async cleanup server session (optional)
                page_id = response.get('page_id')
                if page_id:
                    try:
                        self._make_request('DELETE', f'/api/pages/{page_id}')
                    except Exception:
                        pass  # Ignore cleanup errors
            
            total_lines = len(self.cached_text)
            
            # Validate line number
            if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
                line_number = 1
            
            self.current_line = line_number
            
            # Display page content
            result = self._print_window(self.current_line, self.window, ignore_window=False)
            result['message'] = f"Successfully navigated to {url} and displayed the content."
            return result
            
        except Exception as e:
            # Clear state even on exception
            self._clear_page_state()
            return self._output_error(f'Error navigating to {url}: {str(e)}')
    
    def goto_line(self, line_number: int) -> Dict[str, Any]:
        """Jump to a specific line in the current page.
        
        Args:
            line_number: int: The line number to jump to.
            
        Returns:
            Dict[str, Any]: Dictionary containing page content and status information.
        """
        if not self._check_current_page():
            return self._output_error('No page loaded. Use the `web_browse` first.')
        
        total_lines = len(self.cached_text)
        
        if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
            return self._output_error(f'Line number must be between 1 and {total_lines}.')
        
        self.current_line = self._clamp(line_number, 1, total_lines)
        
        result = self._print_window(self.current_line, self.window, ignore_window=False)
        result['message'] = f"Navigate to line {self.current_line} of the current page {self.current_url}."
        return result
    
    def scroll_down(self) -> Dict[str, Any]:
        """Scroll down the current page.
        
        It will scroll down 100 lines of content.
        
        Returns:
            Dict[str, Any]: Dictionary containing page content and status information.
        """
        if not self._check_current_page():
            return self._output_error('No page loaded. Use the `web_browse` first.')
        
        total_lines = len(self.cached_text)
        new_line = self.current_line + self.window
        
        self.current_line = self._clamp(new_line, 1, total_lines)
        
        result = self._print_window(self.current_line, self.window, ignore_window=True)
        result['message'] = f"Scroll down to line {self.current_line} of the current page {self.current_url} successfully."
        return result
    
    def scroll_up(self) -> Dict[str, Any]:
        """Scroll up the current page.
        
        It will scroll up 100 lines of content.
        
        Returns:
            Dict[str, Any]: Dictionary containing page content and status information.
        """
        if not self._check_current_page():
            return self._output_error('No page loaded. Use the `web_browse` first.')
        
        total_lines = len(self.cached_text)
        new_line = self.current_line - self.window
        
        self.current_line = self._clamp(new_line, 1, total_lines)
        
        result = self._print_window(self.current_line, self.window, ignore_window=True)
        result['message'] = f"Scroll up to line {self.current_line} of the current page {self.current_url} successfully."
        return result

    def search(self, keyword: str, context_lines: int = 5) -> Dict[str, Any]:
        """Search for a keyword in the current page.
        
        It will search for the keyword in the current page and display the context lines around the place where the keyword appear for the first time.
        
        Args:
            keyword: str: The keyword to search for in the current page.
            context_lines: int: The number of context lines to show around each match. Default is 5.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and status information.
        """
        if not self._check_current_page():
            return self._output_error('No page loaded. Use the `web_browse` first.')
        
        # Clear previous search results
        self.search_results = []
        self.keyword = ""
        self.current_search_index = 0
        
        # Find all matches
        for i, line in enumerate(self.cached_text, 1):
            if keyword.lower() in line.lower():
                self.search_results.append({
                    'line_number': i,
                    'content': line.strip(),
                    'context_start': max(1, i - context_lines),
                    'context_end': min(len(self.cached_text), i + context_lines)
                })
                if self.keyword == "":
                    self.keyword = keyword
        
        if not self.search_results:
            return {
                "success": True,
                "message": f'No matches found for "{keyword}" in current page {self.current_url}.',
                "keyword": keyword,
                "total_matches": 0,
                "current_match": 0,
                "current_line": 0,
                "context": [],
                "output": "",
            }
        
        # Return context for first result
        first_result = self.search_results[0]
        context_lines_output = []
        
        for line_num in range(first_result['context_start'], first_result['context_end'] + 1):
            line_content = self.cached_text[line_num - 1] if line_num <= len(self.cached_text) else ''
            marker = '>>>' if line_num == first_result['line_number'] else '   '
            context_lines_output.append(f'{marker}{line_num}|{line_content}')
        
        output_lines = [
            f'[Found {len(self.search_results)} matches for "{keyword}" in current page]',
            f'[Showing context for match 1/{len(self.search_results)} at line {first_result["line_number"]}]'
        ]
        output_lines.extend(context_lines_output)
        
        if len(self.search_results) > 1:
            output_lines.append('[Use `web_page_search_next` to go to the next match]')
        
        # Build complete output and truncate
        full_output = '\n'.join(output_lines)
        full_output = self._truncate_string(full_output)
        
        return {
            "success": True,
            "message": f'Found {len(self.search_results)} matches for "{keyword}" in current page {self.current_url}.',
            "keyword": keyword,
            "total_matches": len(self.search_results),
            "current_match": 1,
            "current_line": first_result['line_number'],
            "context": context_lines_output,
            "output": full_output
        }
    
    def search_next(self, context_lines: int = 5, search_index: int = None) -> Dict[str, Any]:
        """Jump to the place where the keyword appears in the last opened web page.
        
        If search_index is bigger than the number of matches, it will jump to the (search_index % the number of matches) th match.
        
        Args:
            context_lines: int: The number of context lines to show around the match. Default is 5.
            search_index: int: The index of the search result to jump to. If None, jumps to the next result.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and status information.
        """
        if not self.search_results:
            return self._output_error('No search results available. Use `web_page_search` first.')
        
        if search_index is not None:
            self.current_search_index = search_index % len(self.search_results)
        else:
            # Move to next result
           self.current_search_index = (self.current_search_index + 1) % len(self.search_results)
        current_result = self.search_results[self.current_search_index]
        
        # Generate context
        context_lines_output = []
        for line_num in range(current_result['context_start'], current_result['context_end'] + 1):
            line_content = self.cached_text[line_num - 1] if line_num <= len(self.cached_text) else ''
            marker = '>>>' if line_num == current_result['line_number'] else '   '
            context_lines_output.append(f'{marker}{line_num}|{line_content}')
        
        match_number = self.current_search_index + 1
        output_lines = [
            f'[Showing context for match {match_number}/{len(self.search_results)} at line {current_result["line_number"]}]'
        ]
        output_lines.extend(context_lines_output)
        
        if match_number < len(self.search_results):
            output_lines.append('[Use `web_page_search_next` to go to the next match]')
        else:
            output_lines.append('[This is the last match. Use `web_page_search_next` to cycle back to the first match]')
        
        # Build complete output and truncate
        full_output = '\n'.join(output_lines)
        full_output = self._truncate_string(full_output)
        
        return {
            "success": True,
            "message": f'Jump to the {match_number}-th match for "{self.keyword}" in current page {self.current_url}.',
            "keyword": self.keyword,
            "total_matches": len(self.search_results),
            "current_match": match_number,
            "current_line": current_result['line_number'],
            "context": context_lines_output,
            "output": full_output
        }
    
    def get_links(self, page_size: int = 10, page_number: int = 1) -> Dict[str, Any]:
        """Get hyperlinks from the current page.
        
        Args:
            page_size: int: The number of links to show per page. Default is 10.
            page_number: int: The page number to display. Default is 1.
            
        Returns:
            Dict[str, Any]: Dictionary containing link list and status information.
        """
        if not self._check_current_page():
            return self._output_error('No page loaded. Use the `web_browse` first.')
        
        total_links = len(self.cached_links)
        
        if total_links == 0:
            return {
                "success": True,
                "message": f"No links found on current page {self.current_url}.",
                "total_links": 0,
                "page_number": 0,
                "total_pages": 0,
                "page_size": 0,
                "links": [],
                "output": "",
            }
        
        # Calculate pagination
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, total_links)
        total_pages = (total_links + page_size - 1) // page_size
        
        if start_idx >= total_links:
            return self._output_error(f'Page number {page_number} out of range. Total pages: {total_pages}')
        
        # Get links for current page
        current_page_links = self.cached_links[start_idx:end_idx]
        
        # Format output
        output_lines = [
            f'[Found {total_links} links on current page]',
            f'[Showing page {page_number}/{total_pages} ({len(current_page_links)} links)]'
        ]
        
        links_output = []
        for i, link in enumerate(current_page_links, start_idx + 1):
            link_info = f'{i}. {link["text"][:80]}{"..." if len(link["text"]) > 80 else ""}'
            if link["href"]:
                link_info += f' -> {link["href"]}'
            # Add position information
            if link.get("line_number"):
                link_info += f' (at line {link["line_number"]})'
            output_lines.append(link_info)
            links_output.append({
                "index": i,
                "text": link["text"],
                "href": link["href"],
                "line_number": link.get("line_number")
            })
        
        if page_number < total_pages:
            output_lines.append(f'[Use `web_page_get_links(page_number={page_number + 1})` to view more links]')
        
        return {
            "success": True,
            "message": f'Found {total_links} links on current page {self.current_url}.',
            "total_links": total_links,
            "page_number": page_number,
            "total_pages": total_pages,
            "page_size": page_size,
            "links": links_output,
            "output": '\n'.join(output_lines)
        }

    def _truncate_string(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate the string to prevent it from being too long"""
        if max_length is None:
            max_length = self.max_output_length
            
        if len(text) <= max_length:
            return text
            
        # Keep front 50% and back 50% of content
        keep_front = int(max_length * 0.5)
        keep_back = max_length - keep_front - 50  # Reserve space for ellipsis
        
        # Calculate number of omitted characters
        omitted_chars = len(text) - keep_front - keep_back
        
        # Calculate number of omitted lines
        front_text = text[:keep_front]
        back_text = text[-keep_back:]
        middle_text = text[keep_front:-keep_back]
        
        # Calculate number of lines in middle section
        omitted_lines = middle_text.count('\n')
        
        truncated = front_text
        truncated += f"... (The web page's output is too long, omitted {omitted_chars} characters and {omitted_lines} lines in the middle) ..."
        truncated += back_text
        
        return truncated


if __name__ == "__main__":
    # Test code
    web_ops = WebOperations(
        server_host='45.78.231.212', 
        server_port=8124,
        cache_duration_days=1
    )
    
    # Test navigation
    result = web_ops.goto('https://arxiv.org/pdf/2507.15846')
    print("\033[31m" + result.get("output", str(result)) + "\033[0m")
    

    result = web_ops.goto('https://modelscope.cn/models/agentica-org/DeepScaleR-1.5B-Preview')
    print("\033[31m" + result.get("output", str(result)) + "\033[0m")
    # Test jumping to line 10
    result = web_ops.goto_line(10)
    print(result.get("output", str(result)))
    
    # Test scrolling down
    result = web_ops.scroll_down()
    print("\033[32m" + result.get("output", str(result)) + "\033[0m")
    
    # Test search
    result = web_ops.search("deepseek")
    print("\033[33m" + result.get("output", str(result)) + "\033[0m")

    result = web_ops.search_next(search_index=5)
    print("\033[32m" + result.get("output", str(result)) + "\033[0m")
    


    # result = web_ops.goto_line(10)
    # print(result.get("output", str(result)))
    
    # # Test scrolling down
    # result = web_ops.scroll_down()
    # print("\033[32m" + result.get("output", str(result)) + "\033[0m")
    
    # # Test search
    # result = web_ops.search("issue")
    # print("\033[33m" + result.get("output", str(result)) + "\033[0m")
    
    # # Test getting links (pagination)
    # result = web_ops.get_links(page_size=5)
    # print("\033[34m" + result.get("output", str(result)) + "\033[0m")
    
    # Close client
    # web_ops.close()