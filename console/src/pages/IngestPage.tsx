import { useEffect, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Alert,
  Button,
  ClipboardCopy,
  Field,
  FieldLabel,
  List,
  ListItem,
  SectionBand,
  SectionBandContent,
  SectionBandDescription,
  SectionBandEyebrow,
  SectionBandHeader,
  SectionBandTitle,
  SectionStack,
  Stack,
  Textarea,
  TextField,
} from "lumen-ui-kit";
import {
  apiFetch,
  fetchBrainsList,
  getSession,
  type BrainRecord,
} from "../lib/api";
import { PageFrame } from "../components/Workbench";

export default function IngestPage() {
  const session = getSession();
  const [text, setText] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<string | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  const [brains, setBrains] = useState<BrainRecord[]>([]);
  const [brainsError, setBrainsError] = useState<string | null>(null);
  const [newBrainId, setNewBrainId] = useState("");
  const [creating, setCreating] = useState(false);
  const [createdPat, setCreatedPat] = useState<string | null>(null);

  useEffect(() => {
    if (!getSession()?.isSystemPat) {
      setBrainsError("Brain list requires system BRAINPAT_TOKEN");
      return;
    }
    fetchBrainsList()
      .then(setBrains)
      .catch(() =>
        setBrainsError("Brain list requires system BRAINPAT_TOKEN"),
      );
  }, []);

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault();
    setIngesting(true);
    setIngestError(null);
    setIngestResult(null);
    setTaskId(null);
    try {
      const res = await apiFetch<{ message: string; task_id: string }>(
        "/ingest/",
        {
          method: "POST",
          body: JSON.stringify({
            data: { data_type: "text", text_data: text },
          }),
        },
      );
      setIngestResult(res.message);
      setTaskId(res.task_id);
      setText("");
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setIngesting(false);
    }
  }

  async function handleCreateBrain(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setBrainsError(null);
    setCreatedPat(null);
    try {
      const res = await apiFetch<BrainRecord>("/system/brains", {
        method: "POST",
        body: JSON.stringify({ brain_id: newBrainId.trim() }),
      });
      setCreatedPat(res.pat ?? null);
      setBrains((prev) => [...prev, res]);
      setNewBrainId("");
    } catch (err) {
      setBrainsError(
        err instanceof Error ? err.message : "Failed to create brain",
      );
    } finally {
      setCreating(false);
    }
  }

  return (
    <PageFrame className="overflow-auto">
      <SectionStack className="max-w-4xl border border-lumen-border">
        <SectionBand tone="accent">
          <SectionBandHeader>
            <SectionBandEyebrow>Pipeline</SectionBandEyebrow>
            <SectionBandTitle>Ingest</SectionBandTitle>
            <SectionBandDescription>
              Submitting into brain{" "}
              <span className="font-mono text-lumen-foreground">
                {session?.brainId}
              </span>
              . Completed jobs appear under Tasks.
            </SectionBandDescription>
          </SectionBandHeader>
        </SectionBand>

        <SectionBand>
          <SectionBandHeader>
            <SectionBandEyebrow>Content</SectionBandEyebrow>
            <SectionBandTitle>Text ingest</SectionBandTitle>
            <SectionBandDescription>
              Free-form text is queued for extraction and storage.
            </SectionBandDescription>
          </SectionBandHeader>
          <SectionBandContent>
            <form onSubmit={handleIngest}>
              <Stack gap="md">
                <Field>
                  <FieldLabel htmlFor="ingest-text">Text to ingest</FieldLabel>
                  <Textarea
                    id="ingest-text"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    rows={8}
                    required
                    placeholder="Emily organized the AI Ethics Meetup in London on March 8, 2024."
                  />
                </Field>
                {ingestError && (
                  <Alert variant="danger" title="Ingest failed">
                    {ingestError}
                  </Alert>
                )}
                {ingestResult && (
                  <Alert variant="success" title="Submitted">
                    {ingestResult}
                    {taskId && (
                      <>
                        {" "}
                        —{" "}
                        <RouterLink
                          to="/tasks"
                          className="underline underline-offset-2"
                        >
                          View task {taskId}
                        </RouterLink>
                      </>
                    )}
                  </Alert>
                )}
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="submit"
                    disabled={!text.trim()}
                    isPending={ingesting}
                    pendingLabel="Submitting…"
                  >
                    Ingest text
                  </Button>
                  <RouterLink
                    to="/tasks"
                    className="inline-flex h-11 items-center border border-lumen-control-border bg-lumen-action-secondary px-4 text-sm font-medium text-lumen-on-action-secondary"
                  >
                    Open tasks
                  </RouterLink>
                </div>
              </Stack>
            </form>
          </SectionBandContent>
        </SectionBand>

        <SectionBand tone="muted">
          <SectionBandHeader>
            <SectionBandEyebrow>Administration</SectionBandEyebrow>
            <SectionBandTitle>Brains</SectionBandTitle>
            <SectionBandDescription>
              Creating brains requires the system BRAINPAT_TOKEN from your
              environment.
            </SectionBandDescription>
          </SectionBandHeader>
          <SectionBandContent>
            <Stack gap="md">
              {brainsError && (
                <Alert variant="warning" title="Brains unavailable">
                  {brainsError}
                </Alert>
              )}

              {brains.length > 0 && (
                <List>
                  {brains.map((b) => (
                    <ListItem key={b.id ?? b.name_key}>
                      <span className="font-mono text-sm text-lumen-foreground">
                        {b.name_key}
                      </span>
                      {b.pat ? (
                        <span className="ml-2 font-mono text-xs text-lumen-muted-foreground">
                          pat {b.pat.slice(0, 8)}…
                        </span>
                      ) : null}
                    </ListItem>
                  ))}
                </List>
              )}

              <form
                onSubmit={handleCreateBrain}
                className="flex flex-wrap items-end gap-2"
              >
                <TextField
                  id="new-brain-id"
                  label="New brain id"
                  value={newBrainId}
                  onChange={(e) => setNewBrainId(e.target.value)}
                  pattern="[a-zA-Z][a-zA-Z0-9]*"
                  placeholder="newBrainId"
                  containerClassName="min-w-[14rem] flex-1"
                />
                <Button
                  type="submit"
                  variant="secondary"
                  disabled={!newBrainId.trim()}
                  isPending={creating}
                  pendingLabel="Creating…"
                >
                  Create brain
                </Button>
              </form>

              {createdPat && (
                <Alert variant="success" title="New brain PAT (copy now)">
                  <ClipboardCopy value={createdPat} label="Copy PAT" />
                </Alert>
              )}
            </Stack>
          </SectionBandContent>
        </SectionBand>
      </SectionStack>
    </PageFrame>
  );
}
