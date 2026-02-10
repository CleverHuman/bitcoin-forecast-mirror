import React, { useState } from 'react';

interface WidgetProps {
  title: string;
  description: string;
  onDelete: () => void;
  onEdit: () => void;
  position: { x: number; y: number };
  onDragStart: (e: React.MouseEvent | React.TouchEvent) => void;
  children?: React.ReactNode;
}

const Widget: React.FC<WidgetProps> = ({
  title,
  description,
  onDelete,
  onEdit,
  onDragStart,
  children,
}) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const toggleDropdown = () => {
    setIsDropdownOpen((prev) => !prev);
  };

  const handleEdit = () => {
    onEdit();
    setIsDropdownOpen(false);
  };

  const handleDelete = () => {
    onDelete();
    setIsDropdownOpen(false);
  };


  return (
    <div
      className='w-64 p-4 border border-gray-500 bg-white rounded-lg shadow-md'
    >
      <div className='flex justify-between'>
        <div className='p-1'>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth="1.5"
            stroke="#808080"
            className="size-6 cursor-move"
            onMouseDown={onDragStart}
            onTouchStart={onDragStart}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
          </svg>
        </div>
        <div className='relative'>
          <button
            onClick={toggleDropdown}
            className='p-1 text-gray-500'
            aria-label='More options'
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="#808080"
              className="size-6"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
          </button>
          {isDropdownOpen && (
            <div className='absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-10'>
              <div className='py-1'>
                <button
                  onClick={handleEdit}
                  className='block px-4 py-2 text-sm text-gray-800 hover:bg-gray-200 w-full text-left'
                >
                  Settings
                </button>
                <button
                  onClick={handleDelete}
                  className='block px-4 py-2 text-sm text-red-600 hover:bg-gray-200 w-full text-left'
                >
                  Delete
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className='mt-6'>
        <h2 className='text-lg text-gray-500 font-semibold'>{title}</h2>
        <p className='text-sm text-gray-500'>{description}</p>
      </div>
      <div className='mt-4 text-gray-500'>{children}</div>
    </div>
  );
};

export default Widget;
