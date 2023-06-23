## How to use json model building

Some examples are here for you to check out. The two main ways are UNet syntax, and basic syntax.
Both can achieve the same things, but UNet syntax may be more legible.

## Basic syntax

The json is composed of 'modules', which must look similar to the following:

{

    "Tag" : "Arbitrary name, not important, but you must have this."
    "Children" : [
        *List of architecture items and modules*
    ]
}


As you can see, you define the childrens list, which you can then pupulate with more modules, or architecture items.
All children are run sequentially from top to bottom.

<b> A basic architecture item looks like this:</b>

{

    "ComponentClass" : "Conv2d",
    "args" : {
        list of fields to pass to constructor
    }
}

Anything placed in the args field goes to the contructor. Classes must be defined in src.utils.find_class_by_name.py, or be a member of torch.nn.

<b> Basic syntax example </b>

{
    
    "Tag" : "Parent",
    
    "Children":[
        #Example of using a nested module inside a module.
        {
            "Tag":"Early",
            "Children" : [
                {
                    "ComponentClass" : "Conv2d",
                    "args" : {
                        "in_channels" : 1,
                        "out_channels" : 32,
                        "kernel_size":2,
                        "groups" : 1
                    }
                },
                {
                    "ComponentClass" : "MaxPool2d",
                    "args" : {
                        "kernel_size":3
                    }
                }
            ]
        },
        {
            "ComponentClass" : "DecoderBlock",
            "args":{
                "channels":[32, 64, 128, 256]
            },
        },
    ]
}


## UNet syntax

Unet syntax is simple. The root map must have "Encoder", "Decoder", and "Middle" defined. Each of those should be a list. Inside these lists, you define a sequential ordering of modules and architecture components. Of course, it is run in the order of Encoider, Decoder, Middle, regardless of what order you write them.

<b> Example of UNet</b>

{

    "Encoder":[
        {
            "ComponentClass" : "Conv2d",
            "args" : {
                "in_channels" : 64,
                "out_channels" : 128,
                "kernel_size" : 2
            },
            "store_out" : "encoder_a"
        },
    ],
    "Middle":[
        {
            "ComponentClass" : "Conv2d",
            "args" : {
                "in_channels" : 512,
                "out_channels" : 512,
                "kernel_size" : 1
            }
        },
    ],
    "Decoder":[
        {
            "ComponentClass" : "ConvTranspose2d",
            "args" : {
                "in_channels" : 512,
                "out_channels" : 256,
                "kernel_size" : 2,
                "stride" : 2
            },
            "forward_in":"encoder_c"
        },
    ]
}


## Passing values
You can express that the output of a component should be saved by adding the save_out tag:


{

    "ComponentClass" : "Conv2d",
    "args" : {
        list of fields to pass to constructor
    },
    "store_out" : "access_name"
}

Then, you can request this value from any other component **on the same level only**.

Here is how you can request:


{

    "ComponentClass" : "Conv2d",
    "args" : {
        list of fields to pass to constructor
    },
    "forward_in" : "access_name"
}


**You can also equest multiple values:**


{

    "ComponentClass" : "Conv2d",
    "args" : {
        list of fields to pass to constructor
    },
    "save_out" : {"a":"access_name", "b":"some_other_variable"}
}
`

Note that **values are passed as a dict parameter to the forward method**. As in the first example, you would have the dict {"access_name" : *whatever that value ends up being*}
The second example is passed with your defined keys.
