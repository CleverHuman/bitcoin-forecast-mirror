import React, { createContext, useState, useContext } from 'react';
import { Blockchain, blockchainGroups } from '@/components/blockchain-switcher';

interface BlockchainContextType {
  selectedBlockchain: Blockchain;
  setSelectedBlockchain: (blockchain: Blockchain) => void;
}

const BlockchainContext = createContext<BlockchainContextType | undefined>(undefined);

export const BlockchainProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedBlockchain, setSelectedBlockchain] = useState<Blockchain>(blockchainGroups[0].chains[0]);

  return (
    <BlockchainContext.Provider value={{ selectedBlockchain, setSelectedBlockchain }}>
      {children}
    </BlockchainContext.Provider>
  );
};

export const useBlockchain = () => {
  const context = useContext(BlockchainContext);
  if (context === undefined) {
    throw new Error('useBlockchain must be used within a BlockchainProvider');
  }
  return context;
};