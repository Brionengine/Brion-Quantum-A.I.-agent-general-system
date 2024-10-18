class CloudBitcoinMiner:
    def __init__(self, model):
        self.model = model  # Decision tree or neural network model

    def mine_block(self, block_data):
        block_features = [block_data['Block_Number'], block_data['Timestamp'],
                          block_data['Previous_Hash'], block_data['Difficulty'],
                          block_data['Nonce'], block_data['Transaction_Fees'],
                          block_data['Block_Reward']]
        prediction = self.model.predict([block_features])
        if prediction == 1:
            return "Valid Block! Cloud successfully mined a Bitcoin block."
        else:
            return "Invalid Block! Cloud needs to try again with a different nonce."
