import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchCurrentCoinData } from '@/stores/slices/coingecko/coins-market';
import { RootState } from '@/stores/store';

export default function PriceDisplay() {
  const dispatch = useDispatch();
  const { currentCoinData } = useSelector((state: RootState) => state.coinsMarket);

  useEffect(() => {
    dispatch(fetchCurrentCoinData('bitcoin') as any);
  }, [dispatch]);

  if (!currentCoinData) {
    return <div>Loading...</div>;
  }

  const price = currentCoinData.market_data?.current_price?.usd;
  const priceChange = currentCoinData.market_data?.price_change_24h_in_currency?.usd;
  const percentChange = currentCoinData.market_data?.price_change_percentage_24h;

  const isPositive = percentChange >= 0;

  return (
    <div className="text-foreground w-full h-full flex flex-col justify-center p-2">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-4 lg:gap-6">
        <div className="w-full sm:w-auto">
          <div className="flex items-center gap-2 lg:gap-3">
            <img 
              src="https://bitcoin.org/img/icons/opengraph.png" 
              alt="Bitcoin logo" 
              className="w-5 h-5 sm:w-6 sm:h-6 lg:w-7 lg:h-7" 
            />
            <div>
              <h2 className="text-sm  md:text-md lg:text-lg font-bold">{currentCoinData.name} / USD</h2>
              <p className="text-gray-400 text-xs lg:text-sm">{currentCoinData.symbol.toUpperCase()}USD • BITSTAMP</p>
            </div>
          </div>
        </div>
        <div className="flex flex-col w-full sm:w-auto sm:text-right">
          <div className="text-md lg:text-xl xl:text-2xl font-bold">{price.toLocaleString()} USD</div>
          <div className={`text-xs lg:text-sm ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
            {isPositive ? '+' : ''}{priceChange.toLocaleString()} {isPositive ? '+' : ''}{percentChange.toFixed(2)}% today
          </div>
        </div>
      </div>
    </div>
  )
}
