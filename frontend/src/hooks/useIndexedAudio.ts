import { get, set } from 'idb-keyval';
import { newId } from '../lib/id';

export async function saveBlob(blob: Blob): Promise<string> {
  const id = newId();
  await set(id, blob);
  return id;
}

export async function getBlob(id: string): Promise<Blob | undefined> {
  return await get<Blob>(id);
}
