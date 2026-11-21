const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function submitInterviewTurn({ candidateInput, state, signal }) {
  const response = await fetch(`${API_BASE_URL}/v1/interview/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate_input: candidateInput, state }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Interview service rejected the turn (${response.status}).`);
  }

  const result = await response.json();
  if (
    typeof result.reply !== "string" ||
    result.decision == null ||
    result.updated_state == null
  ) {
    throw new Error("Interview service returned an invalid response.");
  }
  return result;
}
