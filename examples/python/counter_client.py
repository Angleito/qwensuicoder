#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sui Move Counter Client - Python SDK Example
"""

import base64
from typing import Optional

from pysui.sui.sui_clients import SyncClient
from pysui.sui.sui_config import SuiConfig
from pysui.sui.sui_builders.move_call import MoveCallTransaction
from pysui.sui.sui_builders.get_object import GetObject
from pysui.sui.sui_types.address import SuiAddress

class CounterClient:
    """
    Client for interacting with the Counter contract on Sui blockchain.
    """
    
    def __init__(self, config_path: Optional[str] = None, package_id: str = None):
        """
        Initialize the counter client.
        
        Args:
            config_path: Path to Sui config file (defaults to ~/.sui/sui_config/client.yaml)
            package_id: ID of the deployed counter package
        """
        # Initialize Sui client configuration
        self.config = SuiConfig.from_config_file(config_path) if config_path else SuiConfig.default_config()
        self.client = SyncClient(self.config)
        self.package_id = package_id
        self.counter_module = "counter"
    
    def create_counter(self) -> str:
        """
        Create a new counter object.
        
        Returns:
            ID of the created counter object
        """
        tx = MoveCallTransaction(
            self.client, 
            signer=self.config.active_address,
            package_object_id=self.package_id,
            module=self.counter_module,
            function="create_and_transfer",
            arguments=[],
            type_arguments=[]
        )
        
        result = tx.execute()
        
        if not result.is_ok():
            raise Exception(f"Failed to create counter: {result.result_string}")
            
        # Extract the created object ID from the transaction results
        effects = result.result_data.effects
        created = effects.created
        
        if not created:
            raise Exception("No objects were created in the transaction")
            
        return created[0].reference.object_id
        
    def increment_counter(self, counter_id: str) -> None:
        """
        Increment a counter by 1.
        
        Args:
            counter_id: ID of the counter object
        """
        tx = MoveCallTransaction(
            self.client, 
            signer=self.config.active_address,
            package_object_id=self.package_id,
            module=self.counter_module,
            function="increment",
            arguments=[counter_id],
            type_arguments=[]
        )
        
        result = tx.execute()
        
        if not result.is_ok():
            raise Exception(f"Failed to increment counter: {result.result_string}")
    
    def increment_counter_by(self, counter_id: str, amount: int) -> None:
        """
        Increment a counter by a custom amount.
        
        Args:
            counter_id: ID of the counter object
            amount: Amount to increment by
        """
        tx = MoveCallTransaction(
            self.client, 
            signer=self.config.active_address,
            package_object_id=self.package_id,
            module=self.counter_module,
            function="increment_by",
            arguments=[counter_id, amount],
            type_arguments=[]
        )
        
        result = tx.execute()
        
        if not result.is_ok():
            raise Exception(f"Failed to increment counter: {result.result_string}")
    
    def reset_counter(self, counter_id: str) -> None:
        """
        Reset a counter to 0.
        
        Args:
            counter_id: ID of the counter object
        """
        tx = MoveCallTransaction(
            self.client, 
            signer=self.config.active_address,
            package_object_id=self.package_id,
            module=self.counter_module,
            function="reset",
            arguments=[counter_id],
            type_arguments=[]
        )
        
        result = tx.execute()
        
        if not result.is_ok():
            raise Exception(f"Failed to reset counter: {result.result_string}")
    
    def get_counter_value(self, counter_id: str) -> int:
        """
        Get the current value of a counter.
        
        Args:
            counter_id: ID of the counter object
            
        Returns:
            Current value of the counter
        """
        # Get the object data
        get_obj = GetObject(self.client, object_id=counter_id)
        result = get_obj.execute()
        
        if not result.is_ok():
            raise Exception(f"Failed to get counter object: {result.result_string}")
        
        # Extract the value field from the object data
        data = result.result_data
        fields = data.content.fields
        return int(fields.value)


def main():
    """
    Example usage of CounterClient.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python counter_client.py PACKAGE_ID [COUNTER_ID]")
        sys.exit(1)
    
    package_id = sys.argv[1]
    counter_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Initialize the counter client
        client = CounterClient(package_id=package_id)
        
        # Create a new counter or use an existing one
        if not counter_id:
            print("Creating a new counter...")
            counter_id = client.create_counter()
            print(f"Created counter with ID: {counter_id}")
        
        # Get initial counter value
        value = client.get_counter_value(counter_id)
        print(f"Initial counter value: {value}")
        
        # Increment counter
        print("Incrementing counter...")
        client.increment_counter(counter_id)
        
        # Get updated value
        value = client.get_counter_value(counter_id)
        print(f"Counter value after increment: {value}")
        
        # Increment by custom amount
        increment_amount = 5
        print(f"Incrementing counter by {increment_amount}...")
        client.increment_counter_by(counter_id, increment_amount)
        
        # Get updated value
        value = client.get_counter_value(counter_id)
        print(f"Counter value after increment by {increment_amount}: {value}")
        
        # Reset counter
        print("Resetting counter...")
        client.reset_counter(counter_id)
        
        # Get updated value
        value = client.get_counter_value(counter_id)
        print(f"Counter value after reset: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 