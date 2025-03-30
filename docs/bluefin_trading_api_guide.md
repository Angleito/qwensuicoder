# Bluefin Trading API Guide for Automated Trading

Generated on Sat Mar 29 22:12:47 PDT 2025

---

# Comprehensive Guide to Automated Trading with Bluefin TS API

## Introduction to Bluefin API for Algorithmic Trading

Bluefin is a financial trading platform that provides APIs for automated trading. This guide will walk you through the setup, authentication, basic trading operations, implementing trading strategies, WebSocket connections, and risk management techniques.

### 1. Setup and Authentication

#### 1.1 Installation
First, ensure you have Node.js installed on your system. You can install it from [Node Package Manager (npm)](https://www.npmjs.com/).

```bash
npm install bluefin-ts-api
```

#### 1.2 Importing the API
In your TypeScript project, import the Bluefin TS API module.

```typescript
import { BluefinAPI } from 'bluefin-ts-api';
```

#### 1.3 Setting Up the Client
Create an instance of the `BluefinAPI` class and provide your access token.

```typescript
const bluefin = new BluefinAPI('your_access_token');
```

### 2. Basic Trading Operations

#### 2.1 Initializing the Client
You can initialize the client with optional configuration options.

```typescript
import { Config } from 'bluefin-ts-api';

const config = new Config();
config.apiKey = 'your_api_key';
config.secret = 'your_secret';

const bluefin = new BluefinAPI(config);
```

#### 2.2 Getting Account Information and Balances
You can retrieve account information and balances to understand your trading environment.

```typescript
async function getAccountInfo() {
  const response = await bluefin.getAccountInfo();
  console.log(response);
}

getAccountInfo();
```

#### 2.3 Creating Market and Limit Orders
You can create market and limit orders using the `createOrder` method.

```typescript
async function placeMarketOrder(symbol: string, quantity: number) {
  const response = await bluefin.createOrder({
    symbol,
    side: 'buy',
    type: 'market',
    quantity,
  });

  console.log(response);
}

placeMarketOrder('BTCUSDT', 0.1);
```

#### 2.4 Canceling Orders
You can cancel orders using the `cancelOrder` method.

```typescript
async function cancelOrder(id: string) {
  const response = await bluefin.cancelOrder(id);

  console.log(response);
}

cancelOrder('your_order_id');
```

#### 2.5 Reading Positions
You can read positions to monitor your account holdings.

```typescript
async function getPositions() {
  const response = await bluefin.getPositions();

  console.log(response);
}

getPositions();
```

### 3. Implementing Trading Strategies

#### 3.1 Simple Trailing Stop-loss Strategy
A simple trailing stop-loss strategy is a basic example of how to implement trading strategies in Bluefin.

```typescript
async function setTrailingStopLoss(symbol: string, quantity: number, stopLossPercentage: number) {
  const orderId = await bluefin.createOrder({
    symbol,
    side: 'buy',
    type: 'limit',
    quantity,
    price: bluefin.getPrice(symbol),
  });

  if (orderId) {
    console.log(`Trailing stop-loss set for order ${orderId}`);

    while (true) {
      try {
        const position = await bluefin.getPosition(orderId);
        if (position) {
          const currentPrice = bluefin.getPrice(position.symbol);
          const newStopLoss = currentPrice * (1 - stopLossPercentage / 100);
          console.log(`Current price: ${currentPrice}, New stop-loss: ${newStopLoss}`);

          await bluefin.updateOrder(orderId, { price: newStopLoss });
        } else {
          break;
        }
      } catch (error) {
        console.error('Error updating order:', error);
        break;
      }
    }
  }
}

setTrailingStopLoss('BTCUSDT', 0.1, 5);
```

#### 3.2 Basic Grid Trading Bot
A basic grid trading bot can be implemented to automate buy and sell orders.

```typescript
async function setGridTrading(symbol: string, quantity: number, levels: number) {
  const initialPrice = bluefin.getPrice(symbol);
  const priceStep = (initialPrice * 2) / levels;

  for (let i = 1; i <= levels; i++) {
    const newPrice = initialPrice + i * priceStep;
    await bluefin.createOrder({
      symbol,
      side: 'buy',
      type: 'limit',
      quantity,
      price: newPrice,
    });

    console.log(`Buy order set for ${newPrice}`);

    // Simulate a delay between orders
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
}

setGridTrading('BTCUSDT', 0.1, 5);
```

#### 3.3 Executing Flash Loans for Arbitrage
Flash loans can be used to execute arbitrage opportunities.

```typescript
async function executeFlashLoan(symbol: string, quantity: number) {
  const orderId = await bluefin.createOrder({
    symbol,
    side: 'buy',
    type: 'limit',
    quantity,
    price: bluefin.getPrice(symbol),
  });

  if (orderId) {
    console.log(`Flash loan set for order ${orderId}`);

    // Simulate a delay between orders
    await new Promise(resolve => setTimeout(resolve, 5000));

    const currentPosition = await bluefin.getPosition(orderId);
    const newPrice = currentPosition.price * 1.1; // Example arbitrage opportunity

    await bluefin.updateOrder(orderId, { price: newPrice });

    console.log(`Flash loan executed for order ${orderId}`);
  }
}

executeFlashLoan('BTCUSDT', 0.1);
```

### 4. WebSocket Connections for Real-Time Data

#### 4.1 Setting Up Price Feeds
You can subscribe to price feeds using the `subscribeToPriceFeed` method.

```typescript
async function startPriceFeed(subscription: string) {
  await bluefin.subscribeToPriceFeed({
    subscription,
    onMessage: (message) => {
      console.log(`Received price feed for ${subscription}:`, message);
    },
  });
}

startPriceFeed('ticker/BTCUSDT');
```

#### 4.2 Order Book Monitoring
You can monitor the order book using the `subscribeToOrderBook` method.

```typescript
async function startOrderBook(subscription: string) {
  await bluefin.subscribeToOrderBook({
    subscription,
    onMessage: (message) => {
      console.log(`Received order book for ${subscription}:`, message);
    },
  });
}

startOrderBook('orderbook/BTCUSDT');
```

#### 4.3 Position Updates
You can subscribe to position updates using the `subscribeToPositionUpdates` method.

```typescript
async function startPositionUpdates(subscription: string) {
  await bluefin.subscribeToPositionUpdates({
    subscription,
    onMessage: (message) => {
      console.log(`Received position update for ${subscription}:`, message);
    },
  });
}

startPositionUpdates('position/BTCUSDT');
```

### 5. Risk Management Techniques

#### 5.1 Position Sizing
Position sizing is crucial for managing risk in automated trading.

```typescript
async function setMaxPositionSize(symbol: string, quantity: number) {
  const response = await bluefin.setMaxPositionSize({
    symbol,
    maxQuantity,
  });

  console.log(response);
}

setMaxPositionSize('BTCUSDT', 0.5);
```

#### 5.2 Stop-loss Implementation
Stop-loss implementation is a basic example of how to use stop-loss orders.

```typescript
async function setStopLoss(symbol: string, quantity: number, stopLossPercentage: number) {
  const orderId = await bluefin.createOrder({
    symbol,
    side: 'buy',
    type: 'limit',
    quantity,
    price: bluefin.getPrice(symbol),
  });

  if (orderId) {
    console.log(`Stop-loss set for order ${orderId}`);

    while (true) {
      try {
        const position = await bluefin.getPosition(orderId);
        if (position) {
          const currentPrice = bluefin.getPrice(position.symbol);
          const newStopLoss = currentPrice * (1 - stopLossPercentage / 100);

          await bluefin.updateOrder(orderId, { price: newStopLoss });
        } else {
          break;
        }
      } catch (error) {
        console.error('Error updating order:', error);
        break;
      }
    }
  }
}

setStopLoss('BTCUSDT', 0.1, 5);
```

#### 5.3 Exposure Limits
Exposure limits can be used to manage risk in automated trading.

```typescript
async function setMaxExposure(symbol: string, exposurePercentage: number) {
  const response = await bluefin.setMaxExposure({
    symbol,
    maxExposure,
  });

  console.log(response);
}

setMaxExposure('BTCUSDT', 0.2);
```

By following these steps, you can implement various strategies and techniques for automated trading in cryptocurrencies using the CoinGecko API.

