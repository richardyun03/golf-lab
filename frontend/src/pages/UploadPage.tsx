import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, Loader2, AlertCircle } from "lucide-react";
import { uploadVideo } from "../lib/api";

export default function UploadPage() {
  const navigate = useNavigate();
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!ext || !["mp4", "mov", "avi"].includes(ext)) {
        setError("Unsupported format. Use MP4, MOV, or AVI.");
        return;
      }
      setError(null);
      setUploading(true);
      setProgress("Uploading video...");

      try {
        setProgress("Analyzing swing... this may take a minute");
        const result = await uploadVideo(file);
        navigate(`/analysis/${result.session_id}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Analysis failed");
        setUploading(false);
        setProgress("");
      }
    },
    [navigate]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="max-w-2xl mx-auto px-6 py-24">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-white mb-3">
          Analyze Your Swing
        </h1>
        <p className="text-gray-400 text-lg">
          Upload a golf swing video to get AI-powered phase detection, fault
          analysis, and metrics.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative border-2 border-dashed rounded-2xl p-16 text-center transition-all cursor-pointer ${
          dragging
            ? "border-emerald-500 bg-emerald-500/5"
            : "border-gray-700 hover:border-gray-500 bg-gray-900/50"
        } ${uploading ? "pointer-events-none opacity-60" : ""}`}
        onClick={() => {
          if (uploading) return;
          const input = document.createElement("input");
          input.type = "file";
          input.accept = ".mp4,.mov,.avi";
          input.onchange = () => {
            const file = input.files?.[0];
            if (file) handleFile(file);
          };
          input.click();
        }}
      >
        {uploading ? (
          <div className="flex flex-col items-center gap-4">
            <Loader2 size={48} className="text-emerald-500 animate-spin" />
            <p className="text-gray-300 text-lg">{progress}</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center">
              <Upload size={28} className="text-gray-400" />
            </div>
            <div>
              <p className="text-gray-200 text-lg font-medium">
                Drop your video here or click to browse
              </p>
              <p className="text-gray-500 text-sm mt-1">
                MP4, MOV, or AVI up to 500MB
              </p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-red-400 bg-red-400/10 rounded-lg px-4 py-3">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
