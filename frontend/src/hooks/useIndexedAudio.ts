import { get, set } from 'idb-keyval';

export async function saveBlob(blob: Blob): Promise<string> {
  const id = crypto.randomUUID();
  await set(id, blob);
  return id;
}

export async function getBlob(id: string): Promise<Blob | undefined> {
  return await get<Blob>(id);
}
