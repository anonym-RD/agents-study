from dataclasses import dataclass
from typing import List, Optional
import time 

@dataclass
class TokenInfo: 
    token: str
    remaining: Optional[int] = None
    reset: Optional[int] = None

class TokenManager: 
    
    def __init__(self, tokens: List[str]): 
        self.tokens_info = [TokenInfo(token=t) for t in tokens]
        self.current_index = 0

    def get_token(self) -> str: 
        token_info = self.tokens_info[self.current_index]
        now = time.time()
        if token_info.remaining == 0 and token_info.reset and now < token_info.reset:
            self.rotate_token()
        return self.tokens_info[self.current_index].token
    
    def rotate_token(self): 
        for _ in range(len(self.tokens_info)):
            self.current_index = (self.current_index + 1) % len(self.tokens_info)
            token_info = self.tokens_info[self.current_index]
            now = time.time()
            if (token_info.remaining is None or token_info.remaining > 50) or (token_info.reset and int(now) > token_info.reset):
                print(f"Switched to token index {self.current_index}")
                return
        raise RuntimeError("All tokens are exhausted until reset!")
    
    def update_limit(self, remaining: int, reset_timestamp: int): 
        self.tokens_info[self.current_index].remaining = remaining
        self.tokens_info[self.current_index].reset = reset_timestamp
    
    def show_index(self) -> int:
        return self.current_index
    
    def get_reset_time(self) -> Optional[int]:
        return self.tokens_info[self.current_index].reset
    
    def get_remaining(self) -> Optional[int]:
        return self.tokens_info[self.current_index].remaining

    def get_all_reset_times(self) -> List[Optional[int]]:
        return [token_info.reset for token_info in self.tokens_info]
    
    def update_index(self, index: int):
        if 0 <= index < len(self.tokens_info):
            self.current_index = index
        else:
            raise IndexError("Token index out of range")