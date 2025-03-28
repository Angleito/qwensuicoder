module counter::counter {
    use sui::object::{Self, UID};
    use sui::transfer;
    use sui::tx_context::{Self, TxContext};

    /// A simple counter object with an integer value
    struct Counter has key, store {
        id: UID,
        value: u64,
    }

    /// Create a new counter with initial value 0
    public fun create(ctx: &mut TxContext): Counter {
        Counter {
            id: object::new(ctx),
            value: 0,
        }
    }

    /// Create and transfer a Counter object to the transaction sender
    public entry fun create_and_transfer(ctx: &mut TxContext) {
        let counter = create(ctx);
        transfer::public_transfer(counter, tx_context::sender(ctx));
    }

    /// Increment counter value by 1
    public entry fun increment(counter: &mut Counter) {
        counter.value = counter.value + 1;
    }

    /// Increment counter value by a custom amount
    public entry fun increment_by(counter: &mut Counter, value: u64) {
        counter.value = counter.value + value;
    }

    /// Get the counter's value
    public fun value(counter: &Counter): u64 {
        counter.value
    }

    /// Reset counter value to 0
    public entry fun reset(counter: &mut Counter) {
        counter.value = 0;
    }
} 