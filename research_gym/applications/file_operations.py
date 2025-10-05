import os
from typing import Optional, List, Dict, Any


class FileOperations:
    """File operations class"""
    
    def __init__(self, workspace_dir: str = "/tmp/workspace", max_output_length: int = 30000):
        """Initialize file operations handler.

        Args:
            workspace_dir: Directory path for file operations. Defaults to "/tmp/workspace".
            max_output_length: Maximum number of characters to output. Defaults to 30000.
        """
        self.workspace_dir = workspace_dir

        self.current_file: Optional[str] = None
        self.current_line: int = 1
        self.window: int = 100

        # output truncation configuration (mainly for file reading operations)
        self.max_output_length = max_output_length  # Maximum output characters

        # ensure the workspace directory exists
        os.makedirs(workspace_dir, exist_ok=True)
        os.chdir(workspace_dir)
    
    def _output_error(self, error_msg: str) -> Dict[str, Any]:
        """Output error message in standardized format.

        This method creates a standardized error response dictionary that maintains
        compatibility with both old and new observation structures.

        Args:
            error_msg: The error message to include in the response.

        Returns:
            Dict containing error information with 'success', 'message', and 'error' fields.
        """
        # for compatibility with the upper observation structure, use message; keep error for compatibility with old callers
        return {"success": False, "message": error_msg, "error": error_msg}
    
    def _check_current_file(self, file_path: Optional[str] = None) -> bool:
        """Check if the current file is valid and exists.

        Args:
            file_path: Optional file path to check. If None, uses the current file.

        Returns:
            bool: True if file exists and is valid, False otherwise.
        """
        if not file_path:
            file_path = self.current_file
        if not file_path or not os.path.isfile(file_path):
            return False
        return True
    
    def _clamp(self, value: int, min_value: int, max_value: int) -> int:
        """Clamp value to specified range.

        Args:
            value: Value to clamp.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.

        Returns:
            int: Clamped value within the specified range.
        """
        return max(min_value, min(value, max_value))
    
    def _print_window(self, file_path: Optional[str], targeted_line: int,
                     window: int, ignore_window: bool = False) -> Dict[str, Any]:
        """Display file content in a window around the target line.

        Args:
            file_path: Optional file path. If None, uses current file.
            targeted_line: Target line number to center the window on.
            window: Number of lines to display before and after target line.
            ignore_window: If True, display entire file regardless of window size.

        Returns:
            Dict containing file content and metadata.
        """
        if not self._check_current_file(file_path) or file_path is None:
            return self._output_error('No file open. Use the open_file function first.')
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
                
                # ensure the content ends with a newline
                if not content.endswith('\n'):
                    content += '\n'
                
                lines = content.splitlines(True)
                total_lines = len(lines)
                
                # adjust the current line
                self.current_line = self._clamp(targeted_line, 1, total_lines)
                half_window = max(1, window // 2)
                
                if ignore_window:
                    # for scroll_down/scroll_up
                    start = max(1, self.current_line)
                    end = min(total_lines, self.current_line + window)
                else:
                    # normal window mode
                    start = max(1, self.current_line - half_window)
                    end = min(total_lines, self.current_line + half_window)
                
                # adjust the range to ensure enough lines are displayed
                if start == 1:
                    end = min(total_lines, start + window - 1)
                if end == total_lines:
                    start = max(1, end - window + 1)
                
                # build the output
                output_lines = []
                
                # file header information
                header = f'[File: {os.path.abspath(file_path)} ({total_lines} lines total)]'
                output_lines.append(header)
                
                # prompt above
                if start > 1:
                    output_lines.append(f'({start - 1} more lines above)')
                else:
                    output_lines.append('(this is the beginning of the file)')
                
                # file content
                file_content = []
                for i in range(start, end + 1):
                    line_content = lines[i - 1].rstrip('\n\r')
                    file_content.append(f'{i}|{line_content}')
                
                # prompt below
                if end < total_lines:
                    output_lines.append(f'({total_lines - end} more lines below)')
                    # add scroll prompt
                    if output_lines[-1].endswith('more lines below)'):
                        output_lines.append('[Use `file_scroll_down` to view the next 100 lines of the file!]')
                else:
                    output_lines.append('(this is the end of the file)')
                
                # build the full output and truncate
                full_output = "\n".join(output_lines[:-1] + file_content + [output_lines[-1]])
                full_output = self._truncate_string(full_output)
                
                return {
                    "success": True,
                    "file_path": os.path.abspath(file_path),
                    "current_line": self.current_line,
                    "total_lines": total_lines,
                    "start_line": start,
                    "end_line": end,
                    "header": header,
                    "content": file_content,
                    "output": full_output
                }
                
        except Exception as e:
            return self._output_error(f'Error reading file: {str(e)}')
    
    def _cur_file_header(self, current_file: Optional[str], total_lines: int) -> str:
        """generate the file header information"""
        if not current_file:
            return ''
        return f'[File: {os.path.abspath(current_file)} ({total_lines} lines total)]'
    
    def open_file(self, path: str, line_number: int = 1, context_lines: Optional[int] = None) -> Dict[str, Any]:
        """Open a file and display its content around a specific line.
        
        The environment will cache the file content for another file action to use until perform next open_file action.
        
        Args:
            path: str: The path to the file to open.
            line_number: int: The line number to focus on (1-indexed). Default is 1.
            context_lines: Optional[int]: Number of lines to show as context. Default is None (uses default window size).
            
        Returns:
            Dict[str, Any]: Dictionary containing file content and status information.
        """
        if not os.path.isfile(path):
            return self._output_error(f'File {path} not found.')
        
        self.current_file = os.path.abspath(path)
        
        # get the total number of lines
        with open(self.current_file, 'r', encoding='utf-8', errors='replace') as file:
            total_lines = max(1, sum(1 for _ in file))
        
        # validate the line number
        if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
            return self._output_error(f'Line number must be between 1 and {total_lines}')
        
        self.current_line = line_number
        
        # set the context lines
        if context_lines is None or context_lines < 1:
            context_lines = self.window
        
        # display the file content
        result = self._print_window(
            self.current_file,
            self.current_line,
            self._clamp(context_lines, 1, 100),
            ignore_window=False
        )
        
        result.update({"message": f"File {self.current_file} opened successfully."})
        return result
    
    def goto_line(self, line_number: int) -> Dict[str, Any]:
        """Jump to a specific line in the currently open file and show the content around the line.
        
        Args:
            line_number: int: The line number to jump to (1-indexed).
            
        Returns:
            Dict[str, Any]: Dictionary containing file content and status information.
        """
        if not self._check_current_file():
            return self._output_error('No file open. Use the open_file function first.')
        
        # get the total number of lines
        with open(str(self.current_file), 'r', encoding='utf-8', errors='replace') as file:
            total_lines = max(1, sum(1 for _ in file))
        
        if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
            return self._output_error(f'Line number must be between 1 and {total_lines}.')
        
        self.current_line = self._clamp(line_number, 1, total_lines)
        
        result = self._print_window(
            self.current_file, 
            self.current_line, 
            self.window, 
            ignore_window=False
        )
        result.update({"message": f"Jump to line {self.current_line} of file {self.current_file} successfully."})
        return result
    
    def scroll_down(self) -> Dict[str, Any]:
        """Scroll down 100 lines in the currently open file.
        
        Returns:
            Dict[str, Any]: Dictionary containing file content and status information.
        """
        if not self._check_current_file():
            return self._output_error('No file open. Use the open_file function first.')
        
        # get the total number of lines
        with open(str(self.current_file), 'r', encoding='utf-8', errors='replace') as file:
            total_lines = max(1, sum(1 for _ in file))
        
        self.current_line = self._clamp(self.current_line + self.window, 1, total_lines)
        
        result = self._print_window(
            self.current_file, 
            self.current_line, 
            self.window, 
            ignore_window=True
        )
        result.update({"message": f"Scroll down to line {self.current_line} of file {self.current_file} successfully."})
        return result
    
    def scroll_up(self) -> Dict[str, Any]:
        """Scroll up 100 lines in the currently open file.
        
        Returns:
            Dict[str, Any]: Dictionary containing file content and status information.
        """
        if not self._check_current_file():
            return self._output_error('No file open. Use the open_file function first.')
        
        # get the total number of lines
        with open(str(self.current_file), 'r', encoding='utf-8', errors='replace') as file:
            total_lines = max(1, sum(1 for _ in file))
        
        self.current_line = self._clamp(self.current_line - self.window, 1, total_lines)
        
        result = self._print_window(
            self.current_file, 
            self.current_line, 
            self.window, 
            ignore_window=True
        )
        result.update({"message": f"Scroll up to line {self.current_line} of file {self.current_file} successfully."})
        return result

    def create_file(self, filename: str, content: str = "") -> Dict[str, Any]:
        """Create a new file with the specified content.
        
        It will also replace the original file if it already exists.
        
        Args:
            filename: str: The name/path of the file to create.
            content: str: The content to write to the new file. Default is empty string.
            
        Returns:
            Dict[str, Any]: Dictionary containing file creation status and information.
        """
        try:
            # create the directory path (if needed)
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            
            # create the file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # add the line number to the added_context
            added_context_with_lines = ""
            if content:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    added_context_with_lines += f"{i}|{line}\n"
                # remove the last extra newline
                if added_context_with_lines.endswith('\n'):
                    added_context_with_lines = added_context_with_lines[:-1]
            
            return {
                "success": True,
                "message": f"File {filename} created successfully",
                "file_path": os.path.abspath(filename),
                "removed_context": "",
                "added_context": added_context_with_lines,
            }
            
        except Exception as e:
            return self._output_error(f'Error creating file: {str(e)}')
    
    def edit_file(self, path: str, start_line: int, end_line: int, content: str) -> Dict[str, Any]:
        """Edit a file given path.
        
        The file's [start,end] lines will be edited to the content. Remember this edit will change the file's 
        line-linenumber index, so do not edit consecutively until you use `read_file` tools to read the new file version.
        
        Args:
            path: str: The path to the file to edit.
            start_line: int: The starting line to be edited (including).
            end_line: int: The ending line to be edited (including).
            content: str: The content to be written or edited in the file. It will replace the content between `start` and `end` lines.
            
        Returns:
            Dict[str, Any]: Dictionary containing edit operation status and information.
        """
        try:
            if not os.path.isfile(path):
                return self._output_error(f'File {path} not found.')
            
            # read the existing content
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # validate the line number range
            if start_line < 1 or start_line > total_lines + 1:
                return self._output_error(f'Start line must be between 1 and {total_lines + 1}')
            if end_line < start_line or end_line > total_lines + 1:
                return self._output_error(f'End line must be between {start_line} and {total_lines + 1}')
            
            # prepare the new content
            new_lines = content.split('\n')
            new_lines = [line + '\n' for line in new_lines]
            
            # handle the case of adding content at the end of the file
            if start_line == total_lines + 1:
                # if the file is not empty and the last line does not end with a newline, add a newline
                if total_lines > 0 and lines and not lines[-1].endswith('\n'):
                    lines[-1] = lines[-1] + '\n'
                # remove the last newline of the new content to avoid duplication
                if new_lines and new_lines[-1].endswith('\n'):
                    new_lines[-1] = new_lines[-1][:-1]
            
            # execute the edit: replace the specified line range
            result_lines = (lines[:start_line-1] + 
                           new_lines + 
                           lines[end_line:])
            
            # write back to the file
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(result_lines)
            
            # add the line number to the removed_context and added_context
            removed_context_with_lines = ""
            if start_line <= total_lines:
                removed_lines = lines[start_line-1:end_line]
                for i, line in enumerate(removed_lines, start_line):
                    removed_context_with_lines += f"{i}|{line}"
                if removed_context_with_lines.endswith('\n'):
                    removed_context_with_lines = removed_context_with_lines[:-1]
            
            added_context_with_lines = ""
            if new_lines:
                for i, line in enumerate(new_lines, start_line):
                    added_context_with_lines += f"{i}|{line}"
                if added_context_with_lines.endswith('\n'):
                    added_context_with_lines = added_context_with_lines[:-1]
            
            return {
                "success": True,
                "message": f"File {path} edited successfully. File updated (edited from line {start_line} to line {end_line}, both including). The line number of the file may changed, please review the changes via `file_read` tool and make sure they are correct (correct indentation, no duplicate lines, etc). Do not edit the file consecutively.",
                "file_path": os.path.abspath(path),
                "new_total_lines": len(result_lines),
                "removed_context": removed_context_with_lines,
                "added_context": added_context_with_lines,
            }
            
        except Exception as e:
            return self._output_error(f'Error editing file: {str(e)}')
    
    def search_dir(self, search_term: str, dir_path: str = './') -> Dict[str, Any]:
        """Search for a text pattern in all files within a directory.
        
        Args:
            search_term: str: The text to search for.
            dir_path: str: The directory path to search in. Default is current directory.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and status information.
        """
        if not os.path.isdir(dir_path):
            return self._output_error(f'Directory {dir_path} not found')
        
        matches = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if search_term in line:
                                matches.append((file_path, line_num, line.strip()))
                except Exception:
                    # skip the files that cannot be read
                    continue
        
        if not matches:
            return {
                "success": True,
                "message": f'No matches found for "{search_term}" in {dir_path}',
                "search_term": search_term,
                "dir_path": dir_path,
                "num_matches": 0,
                "num_files": 0,
                "matches": [],
                "content": [],
                "output": "",
            }
        
        num_matches = len(matches)
        num_files = len(set(match[0] for match in matches))
        
        if num_files > 30:
            return self._output_error(f'More than 30 files ({num_files}) matched for "{search_term}" in {dir_path}. Please narrow your search.')
        
        if num_matches > 100:
            return self._output_error(f'More than 100 ({num_matches}) matched for "{search_term}" in {dir_path}. Please narrow your search.')
        
        # format the output
        output_lines = [f'[Found {num_matches} matches for "{search_term}" in {dir_path}]']
        match_results = []
        
        for file_path, line_num, line in matches:
            match_info = f'{file_path} (Line {line_num}): {line}'
            output_lines.append(match_info)
            match_results.append({
                "file_path": file_path,
                "line": line_num,
                "content": line
            })
        
        output_lines.append(f'[End of matches for "{search_term}" in {dir_path}]')
        
        return {
            "success": True,
            "message": f'Found {num_matches} matches for "{search_term}" in {dir_path}',
            "search_term": search_term,
            "dir_path": dir_path,
            "num_matches": num_matches,
            "num_files": num_files,
            "matches": match_results,
            "content": output_lines,
            "output": '\n'.join(output_lines)
        }
    
    def search_file(self, search_term: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Searches for a text pattern in a specific file or the currently open file.
        
        Args:
            search_term: str: The text to search for.
            file_path: Optional[str]: The file path to search in. If None, searches in currently open file.
                                     Default is None.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and status information.
        """
        if file_path is None:
            file_path = self.current_file
        if file_path is None:
            return self._output_error('No file specified or open. Use the open_file function first.')
        if not os.path.isfile(file_path):
            return self._output_error(f'File {file_path} not found.')
        
        matches = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                for i, line in enumerate(file, 1):
                    if search_term in line:
                        matches.append((i, line.strip()))
        except Exception as e:
            return self._output_error(f'Error reading file: {str(e)}')
        
        if matches:
            output_lines = [f'[Found {len(matches)} matches for "{search_term}" in {file_path}]']
            if len(matches) > 50:
                output_lines[-1] += [" Only present top 50 cases. Please narrow your search."]
            match_results = []
            
            for line_num, line_content in matches[:50]:
                match_info = f'Line {line_num}: {line_content}'
                output_lines.append(match_info)
                match_results.append({
                    "line": line_num,
                    "content": line_content
                })
            
            output_lines.append(f'[End of matches for "{search_term}" in {file_path}]')
            
            return {
                "success": True,
                "message": f'Found {len(matches)} matches for "{search_term}" in {file_path}',
                "file_path": file_path,
                "search_term": search_term,
                "num_matches": len(matches),
                "matches": match_results,
                "content": output_lines,
                "output": '\n'.join(output_lines)
            }
        else:
            return {
                "success": True,
                "message": f'No matches found for "{search_term}" in {file_path}',
                "file_path": file_path,
                "search_term": search_term,
                "num_matches": 0,
                "matches": [],
                "content": [],
                "output": "",
            }
    
    def find_file(self, file_name: str, dir_path: str = './') -> Dict[str, Any]:
        """Finds files by name pattern within a directory.
        
        Args:
            file_name: str: The file name or pattern to search for.
            dir_path: str: The directory path to search in. Default is current directory.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and status information.
        """
        if not os.path.isdir(dir_path):
            return self._output_error(f'Directory {dir_path} not found')
        
        matches = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file_name in file:
                    matches.append(os.path.join(root, file))
        
        if matches:
            output_lines = [f'[Found {len(matches)} matches for "{file_name}" in {dir_path}]']
            if len(matches) > 50:
                output_lines[-1] += "(Too many files, only present first 50 files, Please narrow your search.)"
            for match in matches[:50]:
                output_lines.append(match)
            
            output_lines.append(f'[End of matches for "{file_name}" in {dir_path}]')
            
            return {
                "success": True,
                "message": f'Found {len(matches)} matches for "{file_name}" in {dir_path}',
                "file_path": os.path.abspath(file_name),
                "search_term": file_name,
                "dir_path": dir_path,
                "num_matches": len(matches),
                "matches": matches,
                "content": output_lines,
                "output": '\n'.join(output_lines)
            }
        else:
            return {
                "success": True,
                "message": f'No matches found for "{file_name}" in {dir_path}',
                "file_path": os.path.abspath(file_name),
                "search_term": file_name,
                "dir_path": dir_path,
                "num_matches": 0,
                "matches": [],
                "content": [],
                "output": "",
            }
     
    def list_files(self, path: str = ".", show_hidden: bool = False) -> Dict[str, Any]:
        """List all files and directories in a specified path.
        
        Args:
            path: str: The directory path to list contents of. Default is current directory.
            show_hidden: bool: Whether to show hidden files/directories. Default is False.
            
        Returns:
            Dict[str, Any]: Dictionary containing directory listing and status information.
        """
        if not os.path.isdir(path):
            return self._output_error(f'Directory {path} not found')
        
        try:
            files = []
            directories = []
            
            for item in os.listdir(path):
                if not show_hidden and item.startswith('.'):
                    continue
                
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path):
                    directories.append(item)
            list_directories = sorted(directories)
            list_files = sorted(files)
            # if len(list_directories) > 50:
            #     return self._output_error(f'More than 50 directories ({len(list_directories)}) in {path}. Please narrow your search.')
        
            # if len(list_files) > 100:
            #     return self._output_error(f'More than 100 files ({len(list_files)}) in {dir_path}. Please narrow your search.')
        
            if len(list_directories) > 50:
                list_directories = list_directories[:50]
            if len(list_files) > 50:
                list_files = list_files[:50]
                
            if len(list_directories) < len(directories) or len(list_files) < len(files):
                message = f'Listed {len(list_files)} files (total {len(files)} files) and {len(list_directories)} directories (total {len(directories)} directories) in {path}'
            else:
                message = f'Listed {len(files)} files and {len(directories)} directories in {path}'
            return {
                "success": True,
                "message": message,
                "dir_path": path,
                "list_directories": list_directories,
                "list_files": list_files,
                "total_items": len(files) + len(directories)
            }
            
        except Exception as e:
            return self._output_error(f'Error listing directory: {str(e)}')
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the currently open file.
        
        Returns:
            Dict[str, Any]: Dictionary containing file information and status.
        """
        if not self.current_file:
            return self._output_error("No file is currently open")
        
        try:
            with open(self.current_file, 'r', encoding='utf-8', errors='replace') as f:
                total_lines = sum(1 for _ in f)
            
            return {
                "success": True,
                "message": f"Get file info for {self.current_file} successfully.",
                "file_path": self.current_file,
                "current_line": self.current_line,
                "total_lines": total_lines,
                "window_size": self.window
            }
        except Exception as e:
            return self._output_error(f'Error getting file info: {str(e)}')

    
    def _truncate_string(self, text: str, max_length: Optional[int] = None) -> str:
        """truncate the string to prevent it from being too long"""
        if max_length is None:
            max_length = self.max_output_length
            
        if len(text) <= max_length:
            return text
            
        # keep the front 20% and the back 80% of the content
        keep_front = int(max_length * 0.2)
        keep_back = max_length - keep_front - 50  # Reserve space for ellipsis
        
        # calculate the number of omitted characters
        omitted_chars = len(text) - keep_front - keep_back
        
        # calculate the number of omitted lines
        front_text = text[:keep_front]
        back_text = text[-keep_back:]
        middle_text = text[keep_front:-keep_back]
        
        # calculate the number of lines in the middle part
        omitted_lines = middle_text.count('\n')
        
        truncated = front_text
        truncated += f"... (The file's content is too long, omitted {omitted_chars} characters and {omitted_lines} lines in the middle) ..."
        truncated += back_text
        
        return truncated
