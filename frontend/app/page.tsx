"use client";

import { useState } from "react";

export default function AttireSensePage() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [mode, setMode] = useState<"bg" | "tryon" | "rec">("bg");

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !e.target.files[0]) return;
    const selectedFile = e.target.files[0];

    setFile(selectedFile);
    setImage(URL.createObjectURL(selectedFile));
    setRecommendations([]);
  };

  const handleProcess = async (selectedMode: "bg" | "tryon" | "rec") => {
    setMode(selectedMode);

    if (selectedMode !== "rec" || !file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/recommend", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      const recImages = data.recommendations.map((r: any) => r.image);
      setRecommendations(recImages);

    } catch (err) {
      console.error(err);
    }
  };

  

  return (
    <div className="min-h-screen bg-gray-100">

      {/* HEADER */}
      <div className="w-full bg-black py-4 px-8">
        <h1 className="text-2xl font-semibold text-white">
          AttireSense
        </h1>
      </div>

      {/* MAIN */}
      <div className="px-12 py-14">

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">

          {/* UPLOAD BOX */}
          <label className="h-[420px] border-2 border-dashed border-gray-500 rounded-xl flex items-center justify-center cursor-pointer">
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={handleUpload}
            />

            {image ? (
              <img
                src={image}
                alt="Uploaded"
                className="h-full object-contain p-6"
              />
            ) : (
              <div className="text-gray-500 text-center">
                <p className="text-lg">Click to Upload</p>
                <p className="text-sm">or Drag & Drop</p>
              </div>
            )}
          </label>

          {/* OUTPUT BOX */}
          <div className="relative h-[420px] border-2 border-dashed border-gray-500 rounded-xl flex items-center justify-center overflow-hidden">

            {/* Recommendation Mode */}
            {mode === "rec" && recommendations.length > 0 ? (
              <>
                <div className="flex gap-6 overflow-x-auto p-6 w-full">
                  {recommendations.map((img, index) => (
                    <img
                      key={index}
                      src={img}
                      alt={`Recommendation ${index + 1}`}
                      className="min-w-[200px] h-[300px] object-contain bg-gray-200 rounded-lg"
                    />
                  ))}
                </div>

                {/* Download All */}
                <button
                  onClick={() => {
                    recommendations.forEach((img, index) => {
                      const link = document.createElement("a");
                      link.href = img;
                      link.download = `recommendation_${index + 1}.png`;
                      link.click();
                    });
                  }}
                  className="absolute bottom-4 right-4 bg-gray-900 text-white px-5 py-2 rounded-md hover:bg-black transition"
                >
                  Download All
                </button>
              </>
            ) : result ? (
              <>
                <img
                  src={result}
                  alt="Result"
                  className="h-full object-contain p-6"
                />

                {/* Download Single */}
                <a
                  href={result}
                  download="output.png"
                  className="absolute bottom-4 right-4 bg-gray-900 text-white px-5 py-2 rounded-md hover:bg-black transition"
                >
                  Download
                </a>
              </>
            ) : (
              <span className="text-gray-400 text-lg">
                Inspired styles coming your way
              </span>
            )}
          </div>
        </div>

        {/* BUTTONS */}
        <div className="flex justify-center gap-10 mt-14">

          <button
            onClick={() => handleProcess("bg")}
            disabled={!image}
            className="px-8 py-3 bg-gray-900 text-white rounded-md hover:bg-black transition disabled:opacity-40"
          >
            BACKGROUND REMOVE
          </button>

          <button
            onClick={() => handleProcess("tryon")}
            disabled={!image}
            className="px-8 py-3 bg-gray-900 text-white rounded-md hover:bg-black transition disabled:opacity-40"
          >
            STYLE ME
          </button>

          <button
            onClick={() => handleProcess("rec")}
            disabled={!image}
            className="px-8 py-3 bg-gray-900 text-white rounded-md hover:bg-black transition disabled:opacity-40"
          >
            RECOMMENDATION
          </button>

        </div>
      </div>
    </div>
  );
}
