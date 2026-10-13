# Drafts: reading a message while it is typed

On a machine that reads prompts slowly (a CPU), the wait after pressing send is the time to read
the new message. Most of that message exists before send is pressed: the user is typing it, and the
server is idle meanwhile. A client can send what has been typed so far as a draft. The server reads
it into the context and generates nothing; the request that follows when the user sends the message
finds its prompt already there but for the last tokens.

This costs computing while the user types, and what is read for a text that is then changed is
wasted. It is for one user on one machine, and it is off unless a client asks: nothing here changes
requests that do not use the field.

## The request

On `/v1/chat/completions`, send the conversation as it would be sent, the last message being the
user's unfinished text, with `"draft": true`:

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "first question"},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "I am planning a walking trip in the mount"}
 ],
 "draft": true}
```

- The prompt of a draft is the prompt of the conversation cut where the unfinished text ends: the
  end of the user's turn and the start of the assistant's are not in it.
- The word being typed is left out, with the blanks before it (here the draft is read up to
  `in the`): a half-typed word reads as other tokens than the finished one. So is the last token of
  what remains (`"draft_hold_tokens"`, default 1; 4 after a run of more than 48 bytes without a
  blank, which is kept): a full stop, for one, reads differently once a line break follows it.
- Nothing is generated. The answer comes when the text has been read, with an empty message, the
  usual `timings` (`prompt_n`: tokens read) and `usage` (`prompt_tokens`, and in
  `prompt_tokens_details.cached_tokens` how many of them the context already held).
- Send the other fields the final request will have (`chat_template_kwargs`, tools), or the prompt
  may differ before the user's text and the draft is read for nothing.
- The chat template must put the text of the message into the prompt unchanged. If it does not, the
  request is refused (400) and the client should stop sending drafts.

On `/completion` the same field works on a prompt the client has put together itself: the prompt is
read without its last `draft_hold_tokens` tokens and nothing is generated. The client then has to cut
the unfinished word itself.

## What the server does with the drafts of one message

- A draft that goes on from the last one reads only what is new.
- A draft whose tokens are all in the context already (the same text again, or its end deleted)
  reads nothing and takes nothing back. Only the request that sends the message decides that the
  context has to go back.
- A draft that differs in the middle from what was read (the user edited an earlier word) goes back
  to the last checkpoint before the change and reads from there, like any request. Every draft
  leaves a checkpoint at its end, so that is usually the end of an earlier draft. On models whose
  state cannot be cut at any token (Qwen3.6, Qwen3.8) this is the only way back; the server needs
  `--ctx-checkpoints`.
- A request for the same slot waits while a draft is being read. What the draft reads is the start
  of what that request needs, so the wait is not lost, but a draft is not interrupted: after a long
  paste, the message is answered when the paste has been read.

## For the client

- Send a draft when the user pauses (300 ms or so) or every dozen words, not on every key: small
  batches are the least efficient way to read. A pasted text is worth sending at once.
- Do not send a draft while the answer to the last message is still being generated.
- One draft at a time: wait for the answer to a draft before sending the next, and send the real
  request whenever the user sends, without waiting.
- Measured on a Ryzen 7 5800X with Qwen3.8-Flash-Next (see the project's `RESULTS.md`,
  2026-10-13): a message of about 75 tokens waited 1.1 to 1.4 s for its first token after drafts,
  against 4 to 7 s sent in one piece; a question after a pasted text of 800 tokens waited 2 s.
