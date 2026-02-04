import React from "react";

interface BottomMenuColumnProps {
  icon: React.ReactNode;
  text: string;   
}

interface BottomMenuProps {
  columns: BottomMenuColumnProps[];
}

const BottomMenu: React.FC<BottomMenuProps> = ({ columns }) => {
    return (
      <div className="px-10 flex justify-between gap-x-1 items-center p-4 rounded-md">
        {columns.map((column, index) => (
          <div key={index} className="cursor-pointer flex items-center gap-1">
            {column.icon}
            <span>{column.text}</span>
          </div>
        ))}
      </div>
    );
  };
  

export default BottomMenu;
