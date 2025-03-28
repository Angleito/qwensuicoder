import { TransactionBlock } from '@mysten/sui.js/transactions';
import { SuiClient } from '@mysten/sui.js/client';
import { Ed25519Keypair } from '@mysten/sui.js/keypairs/ed25519';
import { fromB64 } from '@mysten/sui.js/utils';

// Replace these with your values
const PRIVATE_KEY = 'YOUR_PRIVATE_KEY'; // Base64 encoded private key
const PACKAGE_ID = 'YOUR_PACKAGE_ID'; // Counter package ID
const COUNTER_MODULE = 'counter';
const COUNTER_OBJECT_ID = 'YOUR_COUNTER_OBJECT_ID'; // Optional: ID of an existing counter

class CounterClient {
    private client: SuiClient;
    private signer: Ed25519Keypair;
    private packageId: string;
    
    constructor(privateKey: string, packageId: string) {
        // Initialize Sui client
        this.client = new SuiClient({
            url: 'https://fullnode.mainnet.sui.io',
        });
        
        // Initialize signer
        const privateKeyBytes = fromB64(privateKey);
        this.signer = Ed25519Keypair.fromSecretKey(privateKeyBytes);
        
        // Set package ID
        this.packageId = packageId;
    }

    /**
     * Create a new counter object
     */
    async createCounter(): Promise<string> {
        const tx = new TransactionBlock();
        
        // Call the create_and_transfer function
        tx.moveCall({
            target: `${this.packageId}::${COUNTER_MODULE}::create_and_transfer`,
            arguments: [],
        });
        
        // Execute the transaction
        const result = await this.client.signAndExecuteTransactionBlock({
            signer: this.signer,
            transactionBlock: tx,
            options: {
                showEffects: true,
                showObjectChanges: true,
            },
        });
        
        // Find the created counter object ID from transaction effects
        const created = result.effects?.created;
        if (!created || created.length === 0) {
            throw new Error('Failed to create counter object');
        }
        
        // Return the ID of the created counter
        return created[0].reference.objectId;
    }

    /**
     * Increment a counter by 1
     * @param counterId The counter object ID
     */
    async incrementCounter(counterId: string): Promise<void> {
        const tx = new TransactionBlock();
        
        // Call the increment function
        tx.moveCall({
            target: `${this.packageId}::${COUNTER_MODULE}::increment`,
            arguments: [tx.object(counterId)],
        });
        
        // Execute the transaction
        await this.client.signAndExecuteTransactionBlock({
            signer: this.signer,
            transactionBlock: tx,
        });
    }

    /**
     * Increment a counter by a custom amount
     * @param counterId The counter object ID
     * @param amount The amount to increment by
     */
    async incrementCounterBy(counterId: string, amount: number): Promise<void> {
        const tx = new TransactionBlock();
        
        // Call the increment_by function
        tx.moveCall({
            target: `${this.packageId}::${COUNTER_MODULE}::increment_by`,
            arguments: [
                tx.object(counterId),
                tx.pure(amount)
            ],
        });
        
        // Execute the transaction
        await this.client.signAndExecuteTransactionBlock({
            signer: this.signer,
            transactionBlock: tx,
        });
    }

    /**
     * Reset a counter to 0
     * @param counterId The counter object ID
     */
    async resetCounter(counterId: string): Promise<void> {
        const tx = new TransactionBlock();
        
        // Call the reset function
        tx.moveCall({
            target: `${this.packageId}::${COUNTER_MODULE}::reset`,
            arguments: [tx.object(counterId)],
        });
        
        // Execute the transaction
        await this.client.signAndExecuteTransactionBlock({
            signer: this.signer,
            transactionBlock: tx,
        });
    }

    /**
     * Get the counter value
     * @param counterId The counter object ID
     */
    async getCounterValue(counterId: string): Promise<number> {
        // Get object data
        const object = await this.client.getObject({
            id: counterId,
            options: {
                showContent: true,
            },
        });
        
        // Extract the value field
        if (
            object &&
            object.data &&
            object.data.content &&
            'fields' in object.data.content
        ) {
            return Number(object.data.content.fields.value);
        }
        
        throw new Error('Failed to get counter value');
    }
}

// Example usage
async function main() {
    try {
        const counterClient = new CounterClient(PRIVATE_KEY, PACKAGE_ID);
        
        // Create a new counter or use an existing one
        let counterId = COUNTER_OBJECT_ID;
        if (!counterId) {
            console.log('Creating a new counter...');
            counterId = await counterClient.createCounter();
            console.log(`Created counter with ID: ${counterId}`);
        }
        
        // Get initial counter value
        let value = await counterClient.getCounterValue(counterId);
        console.log(`Initial counter value: ${value}`);
        
        // Increment counter
        console.log('Incrementing counter...');
        await counterClient.incrementCounter(counterId);
        
        // Get updated value
        value = await counterClient.getCounterValue(counterId);
        console.log(`Counter value after increment: ${value}`);
        
        // Increment by custom amount
        const incrementAmount = 5;
        console.log(`Incrementing counter by ${incrementAmount}...`);
        await counterClient.incrementCounterBy(counterId, incrementAmount);
        
        // Get updated value
        value = await counterClient.getCounterValue(counterId);
        console.log(`Counter value after increment by ${incrementAmount}: ${value}`);
        
        // Reset counter
        console.log('Resetting counter...');
        await counterClient.resetCounter(counterId);
        
        // Get updated value
        value = await counterClient.getCounterValue(counterId);
        console.log(`Counter value after reset: ${value}`);
        
    } catch (error) {
        console.error('Error:', error);
    }
}

// Uncomment to run the example
// main();

export { CounterClient }; 