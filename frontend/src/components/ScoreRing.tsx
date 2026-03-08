interface Props {
  score: number;
  size?: number;
}

export default function ScoreRing({ score, size = 120 }: Props) {
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const filled = (score / 100) * circumference;

  const color =
    score >= 80
      ? "text-emerald-500"
      : score >= 60
        ? "text-yellow-500"
        : "text-red-500";

  const strokeColor =
    score >= 80
      ? "stroke-emerald-500"
      : score >= 60
        ? "stroke-yellow-500"
        : "stroke-red-500";

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={6}
          className="text-gray-800"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={6}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference - filled}
          className={`${strokeColor} transition-all duration-1000`}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className={`text-3xl font-bold ${color}`}>
          {Math.round(score)}
        </span>
        <span className="text-xs text-gray-500 uppercase tracking-wider">
          Score
        </span>
      </div>
    </div>
  );
}
