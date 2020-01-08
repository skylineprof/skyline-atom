# Skyline for Atom

Skyline is a tool used with [Atom](https://atom.io) to profile, visualize, and
debug the training performance of [PyTorch](https://pytorch.org) neural
networks.

**Note:** Skyline is still under active development and should be considered an
"alpha" product. Its usage and system requirements are subject to change
between versions. See [Versioning](#versioning) for more details.

- [Installing Skyline](#installing-skyline)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Getting Started](#getting-started)
  - [Projects](#projects)
  - [Entry Point](#entry-point)
  - [Example](#example)
- [Providers in Detail](#providers-in-detail)
  - [Model Provider](#model-provider)
  - [Input Provider](#input-provider)
  - [Iteration Provider](#iteration-provider)
- [Versioning](#versioning)
- [Authors](#authors)

-------------------------------------------------------------------------------

## Installing Skyline

### Requirements

Skyline works with GPU-based neural networks that are implemented in PyTorch.
To run Skyline, you need:

- A system equipped with an NVIDIA GPU
- PyTorch 1.1.0+
- Python 3.6+

Skyline is currently only supported on Ubuntu 18.04. It should also work on
other Ubuntu versions that can run Atom and that have Python 3.6+.


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
palette. You can shutdown the interactive profiling session on the command line
by hitting Ctrl-C in your terminal.

**Important:** To analyze your model, Skyline will actually run your code. This
means that when you invoke `skyline interactive`, you need to make sure that
your shell has the proper environments activated (if needed). For example if
you use `virtualenv` to manage your model's dependencies, you need to activate
your `virtualenv` before starting Skyline.


### Projects

To use Skyline, all of the code that you want to profile interactively must be
stored under one common directory. Generally, this just means you need to keep
your own source code under one common directory. Skyline considers all the
files inside this common directory to be part of a *project*, and calls this
common directory your project's *root directory*.

When starting a Skyline interactive profiling session, you must invoke `skyline
interactive <entry point>` inside your project's *root directory*.


### Entry Point

Skyline uses an *entry point* file to learn how to create and train your model.
An entry point file is a regular Python file that contains three top-level
functions:

- `skyline_model_provider`
- `skyline_input_provider`
- `skyline_iteration_provider`

These three functions are called *providers* and must be defined with specific
signatures. The easiest way to understand how to write the providers is to read
through an example.


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
    return (torch.randn((batch_size, 3, 256, 256)).cuda(),)
```


### Providers in Detail

#### Model Provider

```python
skyline_model_provider() -> torch.nn.Module
```

The model provider must take no arguments and return an instance of your model
(a `torch.nn.Module`) that is on the GPU (i.e. you need to call `.cuda()` on
the module before returning it).


#### Input Provider

```python
skyline_input_provider(batch_size: int = 32) -> Tuple
```

The input provider must take a single `batch_size` argument that has a default
value (the batch size you want to profile with). It must return an iterable
(does not *have* to be a `tuple`) that contains the arguments that you would
normally pass to your model's `forward` method. Any `Tensor`s in the returned
iterable must be on the GPU (i.e. you need to call `.cuda()` on them before
returning them).


#### Iteration Provider

```python
skyline_iteration_provider(model: torch.nn.Module) -> Callable
```

The iteration provider must take a single `model` argument, which will be an
instance of your model. This provider must return a callable (e.g., a function)
that, when invoked, runs a single training iteration.


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
