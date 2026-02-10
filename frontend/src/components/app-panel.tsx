import React, { ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface AppPanelProps {
  isExpanded: boolean;
  onToggle: () => void;
  children: ReactNode;
  title?: string;
  className?: string;
  expandedWidth?: string;
  collapsedWidth?: string;
  position?: 'left' | 'right';
}

const AppPanel: React.FC<AppPanelProps> = ({
  isExpanded,
  onToggle,
  children,
  title,
  className = '',
  expandedWidth = 'w-64',
  collapsedWidth = 'w-16',
  position = 'left',
}) => {
  const isLeftPanel = position === 'left';

  return (
    <aside
      className={`h-full transition-[width] duration-300 ease-in-out ${
        isExpanded ? expandedWidth : collapsedWidth
      } bg-background ${className} ${
        isExpanded ? (isLeftPanel ? 'border-r' : 'border-l') : ''
      }`}
    >
      <div className="relative h-full flex flex-col">
        <Button
          onClick={onToggle}
          size='icon'
          variant='outline'
          className={`absolute ${isLeftPanel ? '-right-4' : '-left-4'} top-1/2 z-50 rounded-full h-8 w-8`}
        >
          {isLeftPanel ? (
            <ChevronLeft
              className={`h-3 w-3 ${isExpanded ? '' : 'rotate-180'}`}
            />
          ) : (
            <ChevronRight
              className={`h-3 w-3 ${!isExpanded ? 'rotate-180' : ''}`}
            />
          )}
        </Button>
        <div className="px-3 py-1 flex-shrink-0">
          {title && <h2 className={`text-lg font-semibold ${isExpanded ? '' : 'sr-only'}`}>{title}</h2>}
        </div>
        <div className="flex-grow overflow-y-auto px-1">
          {children}
        </div>
      </div>
    </aside>
  );
};

export default AppPanel;
