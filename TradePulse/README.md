# TradePulse

A financial data analytics platform for algorithmic trading and market analysis.

## Features

- Data ingestion from multiple sources (Alpha Vantage, Yahoo Finance, Polygon)
- Market data processing and analysis
- Trading signal generation
- Backtesting framework
- Sentiment analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TradePulse.git
cd TradePulse

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env to add your API keys
```

## Usage

```python
# Example: Fetch stock data
from TradePulse.data_ingestion.alphavantage_client import AlphaVantageClient
import asyncio

async def get_stock_data():
    client = AlphaVantageClient()
    data = await client.get_intraday("AAPL")
    print(data.head())

# Run the async function
asyncio.run(get_stock_data())
```

## Configuration

The project uses environment variables for configuration. Copy `env.example` to `.env` and add your API keys.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 