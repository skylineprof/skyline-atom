# Skyline for Atom

Skyline is a tool used with [Atom](https://atom.io) to profile, visualize, and
debug the training performance of [PyTorch](https://pytorch.org) neural
networks.

-------------------------------------------------------------------------------

## Installing Skyline

### Requirements

Skyline works with GPU-based neural networks that are implemented in PyTorch.
To run Skyline, you need:

- A system equipped with an NVIDIA GPU
- PyTorch 1.1.0+
- Python 3.6+

Skyline is only supported on Ubuntu 18.04. It should also work on other Ubuntu
versions that can run Atom and that have Python 3.6+.


### Installation

Skyline consists of two components: a command line tool and an Atom plugin
(this repository). Both components must be installed to use Skyline. They can
be installed using `pip` and `apm`:

```
pip install skyline-cli
apm install skyline
```

After installing Skyline, you will be able to invoke the command line tool by
running `skyline` in your shell.


## Getting Started

To use Skyline in your project, you need to first write an *entry point file*,
which is a regular Python file that describes how your model is created and
trained. See the [Entry Point](#entry-point) section for more information.

Once your entry point file is ready, navigate to your project's *root
directory* and run:

```
skyline interactive path/to/entry/point/file
```

Then, open up Atom, execute the `Skyline:Open` command in the command palette
(Ctrl-Shift-P), and hit the "Connect" button that appears on the right.

To shutdown Skyline, execute the `Skyline:Close` command in the command
palette. You can shutdown the profiling session on the command line by hitting
Ctrl-C in your terminal.


## Details

### Projects

Skyline assumes that your code is stored under one top level directory. Skyline
calls this top level directory your project's *root directory*.


### Entry Point


### Example

```
my-project
├── __init__.py
└── main.py
```

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d()

    def forward(self, input):
        return self.conv(input)
```

```python
def skyline_input_provider(batch_size=32):
    return (torch.randn((batch_size, 3, 256, 256)),)
```


## Versioning

Skyline uses semantic versioning. Before the 1.0 release, backwards
compatibility between minor versions will not be guaranteed.

The Skyline command line tool and plugin use *independent* version numbers.
However, it is very likely that minor and major versions of the command line
tool and plugin will be released together (and hence share major/minor version
numbers).

Generally speaking, the most recent version of the command line tool and plugin
will be compatible with each other.


## Authors

Geoffrey Yu <gxyu@cs.toronto.edu>
