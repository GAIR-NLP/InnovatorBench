import requests
import json
import http.client
import time
import os
import hashlib
from typing import Optional, List, Dict, Any


class SearchOperations:
    """Search operations client, supporting multiple search engines"""
    
    def __init__(self, config: Dict[str, Any],
                    cache_dir:str='./search_cache',
                    cache_duration_days:int=1,
                    retry_times:int=30):
        
        # Read search engine type from config
        self.search_engine = config.get('search_engine', '')
        
        # Read corresponding configuration based on search_engine type
        if self.search_engine == 'google':
            self.serper_api_key = config.get('serper_api_key')
            self.azure_bing_search_subscription_key = None
        elif self.search_engine == 'bing':
            self.serper_api_key = None
            self.azure_bing_search_subscription_key = config.get('azure_bing_search_subscription_key')
        else:
            self.serper_api_key = None
            self.azure_bing_search_subscription_key = None
        
        # General configuration
        self.search_max_top_k = config.get('search_max_top_k', 50)  # Maximum search quantity limit
        self.search_region = config.get('search_region', 'us')
        self.search_lang = config.get('search_lang', 'en')
        self.azure_bing_search_mkt = config.get('azure_bing_search_mkt', 'en-US')
        self.retry_times = retry_times
        
        # Cache configuration
        self.cache_dir = cache_dir
        self.cache_duration_days = cache_duration_days
        self.cache_duration_seconds = self.cache_duration_days * 24 * 3600
        
        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Validate configuration
        config_error = self._validate_config()
        if config_error:
            raise ValueError(f"Invalid configuration: {config_error}")
    
    def _get_cache_filename(self, query: str, engine: str, config_hash: str) -> str:
        """Generate cache filename based on query and configuration"""
        cache_key = f"{engine}_{query}_{config_hash}"
        query_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"search_cache_{query_hash}.json")
    
    def _get_config_hash(self) -> str:
        """Generate configuration hash for cache key"""
        # Only use configuration items that affect search results (remove top_k as it's now passed dynamically)
        relevant_config = {
            'search_engine': self.search_engine,
            'search_region': self.search_region,
            'search_lang': self.search_lang,
            'azure_bing_search_mkt': self.azure_bing_search_mkt
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
    
    def _load_from_cache(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Load search results from cache"""
        config_hash = self._get_config_hash()
        cache_file = self._get_cache_filename(query, self.search_engine, config_hash)
        
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
            
            cached_results = cache_data.get('results', [])
            cached_count = cache_data.get('cached_top_k', len(cached_results))
            
            # If requested top_k is greater than cached quantity, return None to trigger new search
            if top_k > cached_count:
                return None
            
            # Return requested quantity of results
            return cached_results[:top_k]
            
        except Exception:
            # Cache file corrupted, delete it
            try:
                os.remove(cache_file)
            except Exception:
                pass
            return None
    
    def _save_to_cache(self, query: str, results: List[Dict[str, Any]], top_k: int) -> None:
        """Save search results to cache"""
        config_hash = self._get_config_hash()
        cache_file = self._get_cache_filename(query, self.search_engine, config_hash)
        
        cache_data = {
            'query': query,
            'engine': self.search_engine,
            'config_hash': config_hash,
            'results': results,
            'cached_top_k': top_k,  # Record the top_k value for this search
            'cache_time': time.time()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save search cache for query '{query}': {e}")
    
    def _output_error(self, error_msg: str, error_code: str = "SEARCH_ERROR") -> Dict[str, Any]:
        """Output error message, including error code"""
        return {
            "success": False,
            "message": error_msg,
            "error_code": error_code
        }
    
    def _validate_config(self) -> Optional[str]:
        """Validate search configuration"""
        if not self.search_engine:
            return "search_engine must be specified"
        
        if self.search_engine not in ['google', 'bing']:
            return f"Unsupported search engine: {self.search_engine}. Supported engines: google, bing"
        
        if self.search_engine == 'google':
            if not self.serper_api_key:
                return "serper_api_key is required for Google search"
        elif self.search_engine == 'bing':
            if not self.azure_bing_search_subscription_key:
                return "azure_bing_search_subscription_key is required for Bing search"
        
        return None
    
    def _serper_google_search(self, query: str, top_k: int, depth: int = 0) -> List[Dict[str, Any]]:
        """Use Serper API for Google search"""
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": self.search_region,
                "hl": self.search_lang,
            })
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            conn.close()

            if not data:
                raise Exception("The Google search API is temporarily unavailable, please try again later.")

            if "organic" not in data:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
            
            # Uniformly format results
            results = []
            for item in data["organic"]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", len(results) + 1),
                    "source": "google_serper"
                }
                # Add additional information (if any)
                if "date" in item:
                    result["date"] = item["date"]
                if "sitelinks" in item:
                    result["sitelinks"] = item["sitelinks"]
                results.append(result)
            
            return results
            
        except Exception as e:
            if depth < self.retry_times:
                time.sleep(1)
                return self._serper_google_search(query, top_k, depth + 1)
            else:
                raise Exception(f"Google search failed after retries: {str(e)}")
    
    def _azure_bing_search(self, query: str, top_k: int, depth: int = 0) -> List[Dict[str, Any]]:
        """Use Azure Bing API for search"""
        params = {'q': query, 'mkt': self.azure_bing_search_mkt, 'count': top_k}
        headers = {'Ocp-Apim-Subscription-Key': self.azure_bing_search_subscription_key}

        try:
            response = requests.get("https://api.bing.microsoft.com/v7.0/search", 
                                  headers=headers, params=params, timeout=30)
            response.raise_for_status()
            json_response = response.json()
            
            if 'webPages' not in json_response or 'value' not in json_response['webPages']:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
            
            # Uniformly format results
            results = []
            for i, item in enumerate(json_response['webPages']['value'], 1):
                result = {
                    "title": item.get('name', ''),
                    "link": item.get('url', ''),
                    "snippet": item.get('snippet', ''),
                    "position": i,
                    "source": "bing_azure"
                }
                # Add additional information (if any)
                if 'dateLastCrawled' in item:
                    result["date_crawled"] = item['dateLastCrawled']
                if 'displayUrl' in item:
                    result["display_url"] = item['displayUrl']
                results.append(result)
            
            return results
            
        except requests.RequestException as e:
            if depth < self.retry_times:
                time.sleep(1)
                return self._azure_bing_search(query, top_k, depth + 1)
            else:
                raise Exception(f"Bing search failed after retries: {str(e)}")
    
    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Search the web for information in google/bing search engine.
        
        Args:
            query: str: The search query to look up on the web.
            top_k: int: The maximum number of search results to return. If the number is bigger than 100, 
                        it will be set to 100. Default is 10.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and status information.
        """
        try:
            # Validate input
            if not query or not query.strip():
                return self._output_error("Search query cannot be empty", "INVALID_QUERY")
            
            if top_k <= 0:
                return self._output_error("top_k must be greater than 0", "INVALID_TOP_K")
            
            if top_k > self.search_max_top_k:
                return self._output_error(f"top_k ({top_k}) exceeds maximum allowed value ({self.search_max_top_k})", "TOP_K_TOO_LARGE")
            
            query = query.strip()
            
            # Check cache
            cached_results = self._load_from_cache(query, top_k)
            if cached_results:
                return {
                    "success": True,
                    "message": f"Top {top_k} search results loaded from cache for query: {query}",
                    "query": query,
                    "requested_top_k": top_k,
                    "total_results": len(cached_results),
                    "search_results": cached_results,
                    "from_cache": True
                }
            
            # Execute search (use maximum top_k to search, get more results for caching)
            if self.search_engine == 'google':
                all_results = self._serper_google_search(query, self.search_max_top_k)
            elif self.search_engine == 'bing':
                all_results = self._azure_bing_search(query, self.search_max_top_k)
            else:
                return self._output_error(f"Unsupported search engine: {self.search_engine}", "INVALID_ENGINE")
            
            # Save to cache (save all results)
            self._save_to_cache(query, all_results, self.search_max_top_k)
            
            # Return requested quantity of results
            results = all_results[:top_k]
            
            return {
                "success": True,
                "message": f"Top {top_k} search results loaded from {self.search_engine} for query: {query}",
                "query": query,
                "requested_top_k": top_k,
                "total_results": len(results),
                "search_results": results,
                "from_cache": False
            }
            
        except Exception as e:
            return self._output_error(f"Search failed: {str(e)}", "SEARCH_FAILED")


def demo_search():
    """Demonstrate search functionality"""
    print("SearchOperations Demo")
    print("=" * 50)
    
    # Demonstrate error handling
    print("\n1. Demonstrate configuration error:")
    try:
        invalid_config = {'search_engine': 'invalid'}
        invalid_ops = SearchOperations(invalid_config)
    except ValueError as e:
        print(f"   Configuration error: {e}")
    
    # Demonstrate invalid query
    print("\n2. Demonstrate invalid query:")
    try:
        # Create a valid configuration but with fake API key
        config = {
            'search_engine': 'google',
            'serper_api_key': 'fake_key_for_demo'
        }
        search_ops = SearchOperations(config)
        result = search_ops.search("", 5)
        print(f"   Query error [{result.get('error_code')}]: {result.get('message')}")
    except ValueError as e:
        print(f"   Configuration error: {e}")
    
    print("\n3. Using example code:")
    try:
        config = {
            'search_engine': 'google',
            'serper_api_key': 'YOUR_SERPER_API_KEY',
            'search_max_top_k': 100,
            'search_region': 'us',
            'search_lang': 'en'
        }
        search_ops = SearchOperations(config)
        result = search_ops.search("assertionerror: failed to get register_center_actor", 20)
        print(result)
        print(result.get("message", str(result)))
    except Exception as e:
        print(f"   Query error: {e}")
        
if __name__ == "__main__":
    demo_search() 