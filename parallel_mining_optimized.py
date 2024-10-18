import threading
import hashlib

def mine_block_thread(nonce_start, nonce_end, previous_hash, transactions, difficulty, result):
    for nonce in range(nonce_start, nonce_end):
        block_contents = str(nonce) + previous_hash + str(transactions)
        block_hash = hashlib.sha256(block_contents.encode()).hexdigest()
        if block_hash.startswith('0' * difficulty):
            result['nonce'] = nonce
            result['hash'] = block_hash
            break

def parallel_mine_block(previous_hash, transactions, difficulty, num_threads=4):
    nonce_range = 1000000  # Adjust as needed
    threads = []
    result = {}
    nonce_step = nonce_range // num_threads

    for i in range(num_threads):
        nonce_start = i * nonce_step
        nonce_end = (i + 1) * nonce_step
        t = threading.Thread(target=mine_block_thread, args=(nonce_start, nonce_end, previous_hash, transactions, difficulty, result))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if 'nonce' in result:
        print(f"Block mined in parallel! Nonce: {result['nonce']}, Hash: {result['hash']}")
    else:
        print("Block not mined within the nonce range.")

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

    # Run parallel mining
    parallel_mine_block(previous_hash, transactions, difficulty)
