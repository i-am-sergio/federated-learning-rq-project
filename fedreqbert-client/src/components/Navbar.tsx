import React, { useState } from "react";
import { IoMdArrowDropdown } from "react-icons/io";
import { useAppContext } from "../context/AppContext";

const models = [
  "Binary Model",
  "Multiclass Model",
];

const Navbar: React.FC = () => {
  const { selectedModel, setSelectedModel } = useAppContext();
  const [isOpenModel, setIsOpenModel] = useState(false); 

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    setIsOpenModel(false);
  };

  return (
    <nav className="bg-transparent p-4 text-white">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center gap-2">
           <span className="font-bold text-2xl tracking-tighter text-violet-400">FedReq</span>
           <span className="font-light text-2xl">BERT</span>
        </div>
        
        <div className="flex items-center">
          {/* Model Selector */}
          <div className="relative">
            <button
              className="text-white font-medium bg-[#27292b] border border-gray-600 rounded-lg hover:bg-[#383a3d] px-4 py-2 transition-all flex items-center text-sm"
              onClick={() => setIsOpenModel(!isOpenModel)}
            >
              <span className="mr-2 text-gray-400">Model:</span>
              {selectedModel}
              <IoMdArrowDropdown className="ml-2" />
            </button>
            {isOpenModel && (
              <div className="absolute right-0 mt-2 w-48 rounded-md shadow-xl bg-[#27292b] border border-gray-600 z-50 overflow-hidden">
                {models.map((model) => (
                  <div
                    key={model}
                    className={`px-4 py-3 text-left text-sm cursor-pointer transition-colors ${selectedModel === model ? 'bg-violet-500/20 text-violet-300' : 'text-white hover:bg-[#4d4f52]'}`}
                    onClick={() => handleModelChange(model)}
                  >
                    {model}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;