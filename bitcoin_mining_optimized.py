import hashlib
import time

def mine_block(previous_hash, transactions, difficulty):
    nonce = 0
    start_time = time.time()
    while True:
        block_contents = str(nonce) + previous_hash + str(transactions)
        block_hash = hashlib.sha256(block_contents.encode()).hexdigest()
        if block_hash.startswith('0' * difficulty):
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Block mined! Nonce: {nonce}, Hash: {block_hash}, Time: {elapsed_time}s")
            return nonce, block_hash
        nonce += 1

if __name__ == "__main__":
    # Previous block hash (for simulation purposes)
    previous_hash = '0000000000000000000b4d0b0d0c0a0b0c0d0e0f0a0b0c0d0e0f0a0b0c0d0e0f'

    # Sample transactions
    transactions = [
        {'from': 'Alice', 'to': 'Bob', 'amount': 1.5},
        {'from': 'Charlie', 'to': 'Dave', 'amount': 2.0},
    ]

    # Difficulty level (number of leading zeros required in hash)
    difficulty = 4  # Adjust difficulty as needed

    # Mine a block
    nonce, new_hash = mine_block(previous_hash, transactions, difficulty)
