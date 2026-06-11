try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    
    # Create a shared limiter instance
    # Use remote address as key (IP based limiting)
    limiter = Limiter(key_func=get_remote_address)
    HAS_SLOWAPI = True

except ImportError:
    HAS_SLOWAPI = False
    
    # Dummy implementation to prevent crashes
    class MockLimiter:
        def limit(self, limit_value):
            # Return a no-op decorator
            def decorator(func):
                return func
            return decorator
            
    limiter = MockLimiter()
    
    def get_remote_address(request):
        return "127.0.0.1"
