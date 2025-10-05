#!/usr/bin/env python3
"""
HTTP Web Server
Run this service on remote machine to provide HTTP API for web operations
No authentication required, simple to use, supports async concurrent processing
"""

import asyncio
import time
import uuid
import json
from quart import Quart, request, jsonify
from typing import Dict, Any, List, Optional
from playwright.async_api import async_playwright
import re


async def load_page_content(url: str, headless: bool = True) -> Dict[str, Any]:
    """Load web page content asynchronously"""
    playwright = None
    browser = None
    page = None
    
    try:
        # Initialize async browser
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        # Create new browser context with user agent
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
        
        page = await context.new_page()
        
        # Hide automation indicators
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        print(f"Attempting to access: {url}")
        
        # Navigate to URL with increased timeout
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            print("Page navigation completed, waiting for network idle...")
            
            # Wait for page load with longer timeout
            await page.wait_for_load_state('networkidle', timeout=45000)
            print("Network idle state achieved")
            
        except Exception as nav_error:
            print(f"Error during navigation or waiting: {nav_error}")
            # Try to get current page content even if error occurred
            await asyncio.sleep(3)  # Wait 3 seconds before trying to get content
        
        # Check if page is still valid
        if page.is_closed():
            raise Exception("Page has been closed, possibly due to anti-bot mechanisms")
        
        # Get page title to verify proper loading
        try:
            title = await page.title()
            print(f"Page title: {title}")
        except Exception:
            title = "Unable to get title"
        
        # Get page text and cache by lines
        try:
            full_text = await page.inner_text('body')
            if not full_text or not full_text.strip():
                # If no text was obtained, try to get HTML content
                full_text = await page.inner_html('body')
                # Simple HTML tag cleanup
                import re
                full_text = re.sub(r'<[^>]+>', ' ', full_text)
                full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Handle cases where full_text might be invalid or empty
            if not full_text or not isinstance(full_text, str):
                cached_text = ["Page content is empty or could not be retrieved"]
            else:
                try:
                    cached_text = full_text.split('\n')
                    # Filter out empty lines at the beginning and end
                    while cached_text and not cached_text[0].strip():
                        cached_text.pop(0)
                    while cached_text and not cached_text[-1].strip():
                        cached_text.pop()
                    
                    # If all lines were empty, provide a default message
                    if not cached_text:
                        cached_text = ["Page appears to be empty. If this is a download link, please use the command operations to download the file instead of using browse operations."]
                except Exception as split_error:
                    print(f"Error splitting text content: {split_error}")
                    cached_text = ["Error processing page content"]
            
        except Exception as text_error:
            print(f"Error getting page text: {text_error}")
            cached_text = [f"Unable to retrieve page text: {str(text_error)}"]
        
        # Get link information
        cached_links = []
        try:
            links = await page.query_selector_all('a')
            print(f"Found {len(links)} links")
            
            for link in links:
                try:
                    if await link.is_visible():
                        text = await link.inner_text()
                        text = text.strip()
                        href = await link.get_attribute('href')
                        if text and len(text) > 0:
                            # Get link context information
                            context = await get_link_context(link, text)
                            
                            # Find link line number in text
                            line_number = find_link_line_number(cached_text, text, context)
                            cached_links.append({
                                'text': text, 
                                'href': href, 
                                'line_number': line_number,
                                'context': context
                            })
                except Exception:
                    # Ignore individual link processing errors
                    continue
                    
        except Exception as links_error:
            print(f"Error getting links: {links_error}")
        
        print(f"Successfully processed page, text lines: {len(cached_text)}, links: {len(cached_links)}")
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'total_lines': len(cached_text),
            'total_links': len(cached_links),
            'cached_text': cached_text,
            'cached_links': cached_links
        }
        
    except Exception as e:
        error_msg = f'Failed to load page {url}: {str(e)}'
        print(f"Error: {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }
    finally:
        # Ensure resources are cleaned up
        try:
            if page and not page.is_closed():
                await page.close()
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
            pass  # Ignore cleanup errors


async def get_link_context(link_element, link_text: str) -> str:
    """Get link context information including parent element and surrounding text"""
    try:
        # Use JavaScript to get context
        context_script = """
        (linkEl) => {
            // Method 1: Get parent element text
            let parent = linkEl.parentElement;
            if (parent && parent.innerText.length > linkEl.innerText.length) {
                return parent.innerText.trim();
            }
            
            // Method 2: Look up for container with enough text
            let current = linkEl.parentElement;
            while (current && current.innerText.length < 100 && current.parentElement) {
                current = current.parentElement;
            }
            
            if (current && current.innerText.length > linkEl.innerText.length) {
                return current.innerText.trim();
            }
            
            // Method 3: Return link text itself
            return linkEl.innerText.trim();
        }
        """
        
        context = await link_element.evaluate(context_script)
        return context[:200] if context else link_text  # Limit length
        
    except Exception:
        return link_text


def find_link_line_number(cached_text: List[str], link_text: str, link_context: str) -> Optional[int]:
    """Find link line number in text using context for precise location"""
    if not cached_text or not link_text:
        return None
    
    # Find all lines containing link text
    matching_lines = []
    for i, line in enumerate(cached_text, 1):
        if link_text in line:
            matching_lines.append(i)
    
    if not matching_lines:
        return None
    
    # If only one match, return directly
    if len(matching_lines) == 1:
        return matching_lines[0]
    
    # If multiple matches, use context information for precise location
    if link_context and len(link_context.strip()) > len(link_text):
        for line_num in matching_lines:
            line = cached_text[line_num - 1]
            # Check if context matches (compare after removing extra whitespace)
            line_clean = ' '.join(line.split())
            context_clean = ' '.join(link_context.split())
            if context_clean in line_clean or line_clean in context_clean:
                return line_num
    
    # If context can't distinguish, return first match
    return matching_lines[0]


class HTTPWebServer:
    """HTTP Web Server"""
    
    def __init__(self, port: int = 8124):
        self.port = port
        self.app = Quart(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup routes"""
        
        @self.app.route('/api/pages', methods=['POST'])
        async def load_page():
            """Load web page asynchronously"""
            try:
                data = await request.get_json() or {}
                url = data.get('url')
                headless = data.get('headless', True)
                
                if not url:
                    return jsonify({
                        'success': False,
                        'error': 'URL is required'
                    }), 400
                
                # Load page content asynchronously, return immediately when done
                result = await load_page_content(url, headless)
                
                if result.get('success'):
                    return jsonify(result)
                else:
                    return jsonify(result), 500
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/health', methods=['GET'])
        async def health_check():
            """Health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time()
            })
        
        @self.app.route('/timestamp', methods=['GET'])
        async def get_timestamp():
            """Get current timestamp"""
            return jsonify({
                'timestamp': time.time()
            })
    
    def start_server(self, host: str = '0.0.0.0', debug: bool = False):
        """Start HTTP server"""
        print(f"Starting HTTP Web Server on {host}:{self.port}")
        print("No authentication required - server is open to all requests")
        print("Make sure to use this only in trusted environments!")
        print("Supporting concurrent async requests")
        
        self.app.run(host=host, port=self.port, debug=debug)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HTTP Web Server (No Authentication, Async)')
    parser.add_argument('--port', '-p', type=int, default=8124, help='Server port (default: 8124)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = HTTPWebServer(port=args.port)
    
    try:
        server.start_server(host=args.host, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == '__main__':
    main() 