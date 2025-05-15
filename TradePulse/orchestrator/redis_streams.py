"""
Redis Streams module for orchestrating data flow.

This module provides functionality for working with Redis Streams,
which are used as the messaging infrastructure for the real-time
data processing pipeline.
"""
import os
import sys
import json
import time
import asyncio
import redis
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

logger = setup_logger('redis_streams')

class RedisStreamClient:
    """Client for working with Redis Streams."""
    
    def __init__(self, redis_url=None):
        """
        Initialize Redis Stream client.
        
        Args:
            redis_url (str, optional): Redis connection URL. If None, reads from config.
        """
        self.redis_url = redis_url or config.REDIS_URL
        self.redis = redis.from_url(self.redis_url)
        self.consumer_groups = {}
        
        logger.info(f"Connected to Redis at {self.redis_url}")
    
    def create_stream(self, stream_name):
        """
        Create a stream if it doesn't exist.
        
        Args:
            stream_name (str): Name of the stream
            
        Returns:
            bool: True if successful
        """
        try:
            # Check if stream exists by trying to get its info
            self.redis.xinfo_stream(stream_name)
            logger.debug(f"Stream {stream_name} already exists")
            return True
        except redis.exceptions.ResponseError:
            # Stream doesn't exist, create it with a minimal entry
            self.redis.xadd(stream_name, {'init': 'true'})
            logger.info(f"Created stream {stream_name}")
            # Delete the initialization entry
            self.redis.xtrim(stream_name, 0)
            return True
        except Exception as e:
            logger.error(f"Error creating stream {stream_name}: {e}")
            return False
    
    def create_consumer_group(self, stream_name, group_name, start_id='0'):
        """
        Create a consumer group for a stream.
        
        Args:
            stream_name (str): Name of the stream
            group_name (str): Name of the consumer group
            start_id (str): ID to start consuming from ('0' for beginning, '$' for new messages)
            
        Returns:
            bool: True if successful
        """
        try:
            # Create the stream if it doesn't exist
            self.create_stream(stream_name)
            
            # Create the consumer group
            try:
                self.redis.xgroup_create(stream_name, group_name, start_id, mkstream=True)
                logger.info(f"Created consumer group {group_name} for stream {stream_name}")
            except redis.exceptions.ResponseError as e:
                if 'BUSYGROUP' in str(e):
                    logger.debug(f"Consumer group {group_name} already exists for stream {stream_name}")
                else:
                    raise
                
            # Store group information
            self.consumer_groups[(stream_name, group_name)] = {
                'stream': stream_name,
                'group': group_name,
                'last_id': '0-0'
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating consumer group {group_name} for stream {stream_name}: {e}")
            return False
    
    def add_message(self, stream_name, message):
        """
        Add a message to a stream.
        
        Args:
            stream_name (str): Name of the stream
            message (dict): Message to add
            
        Returns:
            str: Message ID or None if error
        """
        if not message or not isinstance(message, dict):
            logger.error(f"Invalid message format for stream {stream_name}")
            return None
            
        try:
            # Convert non-string values to strings
            message_copy = {}
            for key, value in message.items():
                if isinstance(value, (dict, list, tuple, set)):
                    message_copy[key] = json.dumps(value)
                else:
                    message_copy[key] = str(value)
                    
            # Add message to stream
            msg_id = self.redis.xadd(stream_name, message_copy)
            return msg_id
            
        except Exception as e:
            logger.error(f"Error adding message to stream {stream_name}: {e}")
            return None
    
    def add_dataframe_row(self, stream_name, df_row):
        """
        Add a pandas DataFrame row to a stream.
        
        Args:
            stream_name (str): Name of the stream
            df_row (pd.Series): DataFrame row to add
            
        Returns:
            str: Message ID or None if error
        """
        if not isinstance(df_row, (pd.Series, dict)):
            logger.error(f"Invalid row format for stream {stream_name}")
            return None
            
        try:
            # Convert Series to dict
            if isinstance(df_row, pd.Series):
                row_dict = df_row.to_dict()
            else:
                row_dict = df_row
                
            # Add to stream
            return self.add_message(stream_name, row_dict)
            
        except Exception as e:
            logger.error(f"Error adding DataFrame row to stream {stream_name}: {e}")
            return None
    
    async def read_messages(self, stream_name, count=10, block=1000):
        """
        Read messages from a stream (non-consumer group).
        
        Args:
            stream_name (str): Name of the stream
            count (int): Maximum number of messages to read
            block (int): Milliseconds to block
            
        Returns:
            list: List of messages
        """
        try:
            # Create stream if it doesn't exist
            self.create_stream(stream_name)
            
            # Read messages
            response = await asyncio.to_thread(
                self.redis.xread,
                {stream_name: '0'},
                count=count,
                block=block
            )
            
            # Process response
            messages = []
            
            if response:
                # Extract messages
                for stream_data in response:
                    stream, stream_messages = stream_data
                    for msg_id, msg_data in stream_messages:
                        # Add message ID
                        msg_data['_id'] = msg_id
                        messages.append(msg_data)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error reading messages from stream {stream_name}: {e}")
            return []
    
    async def read_consumer_group(self, stream_name, group_name, consumer_name, count=10, block=1000):
        """
        Read messages from a consumer group.
        
        Args:
            stream_name (str): Name of the stream
            group_name (str): Name of the consumer group
            consumer_name (str): Name of the consumer
            count (int): Maximum number of messages to read
            block (int): Milliseconds to block
            
        Returns:
            list: List of messages
        """
        try:
            # Ensure consumer group exists
            key = (stream_name, group_name)
            if key not in self.consumer_groups:
                self.create_consumer_group(stream_name, group_name)
            
            # Read messages from group
            response = await asyncio.to_thread(
                self.redis.xreadgroup,
                group_name,
                consumer_name,
                {stream_name: '>'},
                count=count,
                block=block
            )
            
            # Process response
            messages = []
            
            if response:
                # Extract messages
                for stream_data in response:
                    stream, stream_messages = stream_data
                    for msg_id, msg_data in stream_messages:
                        # Add message ID
                        msg_data['_id'] = msg_id
                        messages.append(msg_data)
                        
                        # Update last ID
                        if key in self.consumer_groups:
                            self.consumer_groups[key]['last_id'] = msg_id
            
            return messages
            
        except Exception as e:
            logger.error(f"Error reading from consumer group {group_name} on stream {stream_name}: {e}")
            return []
    
    def acknowledge_message(self, stream_name, group_name, message_id):
        """
        Acknowledge a message in a consumer group.
        
        Args:
            stream_name (str): Name of the stream
            group_name (str): Name of the consumer group
            message_id (str): ID of the message to acknowledge
            
        Returns:
            bool: True if successful
        """
        try:
            self.redis.xack(stream_name, group_name, message_id)
            return True
        except Exception as e:
            logger.error(f"Error acknowledging message {message_id}: {e}")
            return False
    
    def get_pending_messages(self, stream_name, group_name):
        """
        Get pending messages for a consumer group.
        
        Args:
            stream_name (str): Name of the stream
            group_name (str): Name of the consumer group
            
        Returns:
            list: List of pending message IDs
        """
        try:
            # Get pending info
            pending_info = self.redis.xpending(stream_name, group_name)
            
            if pending_info['pending'] > 0:
                # Get detailed pending info
                pending_messages = self.redis.xpending_range(
                    stream_name,
                    group_name,
                    min='-',
                    max='+',
                    count=pending_info['pending']
                )
                
                return pending_messages
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting pending messages for group {group_name}: {e}")
            return []
    
    def claim_message(self, stream_name, group_name, consumer_name, message_id, min_idle_time=60000):
        """
        Claim a pending message.
        
        Args:
            stream_name (str): Name of the stream
            group_name (str): Name of the consumer group
            consumer_name (str): Name of the consumer
            message_id (str): ID of the message to claim
            min_idle_time (int): Minimum idle time in milliseconds
            
        Returns:
            dict: Message data or None if error
        """
        try:
            # Claim message
            result = self.redis.xclaim(
                stream_name,
                group_name,
                consumer_name,
                min_idle_time,
                message_id
            )
            
            if result and len(result) > 0:
                msg_id, msg_data = result[0]
                msg_data['_id'] = msg_id
                return msg_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error claiming message {message_id}: {e}")
            return None
    
    def messages_to_dataframe(self, messages):
        """
        Convert a list of messages to a pandas DataFrame.
        
        Args:
            messages (list): List of message dictionaries
            
        Returns:
            pd.DataFrame: DataFrame of messages
        """
        if not messages:
            return pd.DataFrame()
            
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(messages)
        
        # Convert JSON strings back to Python objects
        for col in df.columns:
            # Skip ID column
            if col == '_id':
                continue
                
            # Try to parse JSON
            try:
                df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) and x.startswith('{') else x)
            except:
                pass
        
        return df

async def consume_stream(redis_client, stream_name, group_name, consumer_name, 
                       batch_size=10, process_func=None, sleep_time=0.1):
    """
    Continuously consume messages from a stream.
    
    Args:
        redis_client (RedisStreamClient): Redis stream client
        stream_name (str): Name of the stream
        group_name (str): Name of the consumer group
        consumer_name (str): Name of the consumer
        batch_size (int): Number of messages to read at once
        process_func (callable): Function to process messages
        sleep_time (float): Time to sleep between polling
    """
    if not process_func:
        logger.error("No process function provided")
        return
        
    # Create consumer group
    redis_client.create_consumer_group(stream_name, group_name)
    
    # Main consumption loop
    while True:
        try:
            # Read messages
            messages = await redis_client.read_consumer_group(
                stream_name, group_name, consumer_name, count=batch_size, block=1000
            )
            
            if messages:
                # Process messages
                for message in messages:
                    # Extract message ID
                    msg_id = message.get('_id')
                    if not msg_id:
                        continue
                    
                    try:
                        # Process message
                        await process_func(message)
                        
                        # Acknowledge message
                        redis_client.acknowledge_message(stream_name, group_name, msg_id)
                        
                    except Exception as e:
                        logger.error(f"Error processing message {msg_id}: {e}")
            
            else:
                # No messages, sleep before polling again
                await asyncio.sleep(sleep_time)
            
            # Check for pending messages
            pending = redis_client.get_pending_messages(stream_name, group_name)
            
            for pending_msg in pending:
                # Check if message idle time > 30s
                if pending_msg['idle'] > 30000:
                    # Claim message
                    message = redis_client.claim_message(
                        stream_name, group_name, consumer_name, pending_msg['message_id']
                    )
                    
                    if message:
                        try:
                            # Process message
                            await process_func(message)
                            
                            # Acknowledge message
                            redis_client.acknowledge_message(stream_name, group_name, pending_msg['message_id'])
                            
                        except Exception as e:
                            logger.error(f"Error processing claimed message {pending_msg['message_id']}: {e}")
        
        except asyncio.CancelledError:
            logger.info(f"Consumer {consumer_name} for {stream_name}/{group_name} cancelled")
            break
            
        except Exception as e:
            logger.error(f"Error in consumer loop for {stream_name}/{group_name}/{consumer_name}: {e}")
            await asyncio.sleep(1)  # Sleep longer on error

async def dummy_process(message):
    """
    Dummy message processor for testing.
    
    Args:
        message (dict): Message to process
    """
    print(f"Processing message: {message}")
    await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create Redis client
        redis_client = RedisStreamClient()
        
        # Test streams
        price_stream = config.PRICE_STREAM
        feature_stream = config.FEATURE_STREAM
        
        # Create streams
        redis_client.create_stream(price_stream)
        redis_client.create_stream(feature_stream)
        
        # Create consumer groups
        redis_client.create_consumer_group(price_stream, "price_processors")
        redis_client.create_consumer_group(feature_stream, "feature_processors")
        
        # Add some test messages
        print("Adding test price data...")
        for i in range(5):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            msg_id = redis_client.add_message(price_stream, {
                'symbol': 'AAPL',
                'timestamp': timestamp,
                'price': 150.0 + i,
                'volume': 1000 + i * 100
            })
            print(f"Added message with ID: {msg_id}")
        
        # Read messages without consumer group
        print("\nReading messages directly:")
        messages = await redis_client.read_messages(price_stream, count=10)
        for msg in messages:
            print(f"Message: {msg}")
        
        # Read messages with consumer group
        print("\nReading messages with consumer group:")
        group_messages = await redis_client.read_consumer_group(
            price_stream, "price_processors", "test_consumer", count=10
        )
        for msg in group_messages:
            print(f"Group Message: {msg}")
            # Acknowledge message
            redis_client.acknowledge_message(price_stream, "price_processors", msg['_id'])
        
        # Test continuous consumption (run for 10 seconds)
        print("\nStarting continuous consumption (will run for 10 seconds)...")
        
        # Create task
        consumer_task = asyncio.create_task(
            consume_stream(
                redis_client, 
                price_stream, 
                "price_processors", 
                "test_consumer_2",
                process_func=dummy_process
            )
        )
        
        # Add messages while consuming
        print("Adding more messages during consumption...")
        for i in range(20):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            redis_client.add_message(price_stream, {
                'symbol': 'MSFT',
                'timestamp': timestamp,
                'price': 250.0 + i,
                'volume': 2000 + i * 100
            })
            await asyncio.sleep(0.5)  # Add a message every 0.5 seconds
        
        # Let consumer run for a bit
        await asyncio.sleep(5)
        
        # Cancel consumer task
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            print("Consumer task cancelled")
        
        print("Example completed")
    
    asyncio.run(main()) 