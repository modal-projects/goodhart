# goodhart

Stop reward hacking dead in its tracks.

Built on [Modal](https://gym.modal.dev/) and [Jev](https://typesafe.ai/).

## Use it

Auth with [Modal](https://modal.com/):

```bash
modal setup
```

Set up the [dashboard](https://gym.modal.dev/guides/dashboard):

```bash
training-gym setup
```

Create a [Modal Secret](https://modal.com/docs/guide/secrets) with your [Typesafe API key](https://console.typesafe.ai/keys):

```bash
modal secret create typesafe-secret TYPESAFE_API_KEY=<api-key>
```

Run the script:

```bash
uv run main.py
```
