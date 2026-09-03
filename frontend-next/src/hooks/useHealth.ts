import { useCallback, useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "../services/api";

export function useHealth(pollMs = 8000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
      setError(null);
    } catch (e) {
      setHealth(null);
      setError(e instanceof Error ? e.message : "health check failed");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), pollMs);
    return () => window.clearInterval(id);
  }, [refresh, pollMs]);

  return { health, error, refresh };
}
