# Dialog Content Loaders

In dialog-lib, we support loading dialog content from various sources. This is done by calling a `loader` function or using our CLI.

## Supported contents:

 - CSV
 - Google Sheets
 - Web - Available soon

### Using our CLI

You can use our CLI to load content from the above listed sources. The CLI is available as a global package and is installed whenever you have dialog-lib installed.

In order to install dialog-lib, run the following command:

```bash
pip install dialog-lib
```

Once you have dialog-lib installed, you can use the CLI to load content from the above listed sources. The CLI is available as a global package and is installed whenever you have dialog-lib installed.

Here is a quick preview on commands available in the CLI:

```bash
$ dialog

Usage: dialog [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  anthropic
  load-csv
  load-google-sheets
  openai
```

### Using the loader command

#### Loading a CSV file

To load a CSV file, you can use the `load-csv` command. Here is an example:

```bash
$ dialog load-csv --file my-amazing-file.csv
```

#### Loading a Google Sheet file

To load a Google Sheet file, you can use the `load-google-sheets` command. Here is an example:

```bash

$ dialog load-google-sheets --spreadsheet-url https://docs.google.com/spreadsheets/d/MY-SPREADSHEET-URL-HERE/ --sheet-name Sheet1 --credentials-path /my/credentials/path/here.json
```

The credentials path must be the full path of a Service Account JSON file. You can create a Service Account JSON file by following the instructions [here](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).