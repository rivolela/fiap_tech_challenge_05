#!/usr/bin/env python3
import os
import hmac
import hashlib

def hash_api_key(api_key):
    salt = "default-salt-change-in-production"
    h = hmac.new(
        salt.encode('utf-8'),
        api_key.encode('utf-8'),
        hashlib.sha256
    )
    return h.hexdigest()

# Test with our key
key = "fiap-api-key"
print(f"Original key: {key}")
print(f"Hashed key: {hash_api_key(key)}")

# Print the hardcoded keys from security.py for comparison
print("\nHardcoded keys in security.py:")
print("526ad77089d41f0b24c9c4dbdb1d861173a0b7d12b5da3148ca86c3ae56cd75c: admin (your-api-key)")
print("074b4cc16ac5a29907bc44f4abf13e5158363416ce10d2cc77fb12252d242ffa: admin (fiap-api-key)")
print("ceb1aaa0d16c8851422baa230eed00417def9c13cb7dfff0c55f257a77dcae9b: read-only (test-api-key)")