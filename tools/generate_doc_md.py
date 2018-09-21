# This script generates mkdocs friendly Markdown documentation from a python package.
# It is based on the the following blog post by Christian Medina
# https://medium.com/python-pandemonium/python-introspection-with-the-inspect-module-2c85d5aa5a48#.twcmlyack 

import pydoc
import os, sys

module_header = "# Package {} Documentation\n"
class_header = "## Class {}"
function_header = "### {}"


def getmarkdown(module, exclude):
    output = [ module_header.format(module.__name__) ]
    
#    if module.__doc__:
#        output.append(module.__doc__)
    #output.extend(getfunctions(module))
    output.extend(getclasses(module, exclude))
    return "\n".join((str(x) for x in output))

def getclasses(item, exclude):
    output = list()
    for cl in pydoc.inspect.getmembers(item, pydoc.inspect.isclass) :
        
        if cl[0] != "__class__" and not cl[0].startswith("_") and (cl[0] not in exclude):
            # Consider anything that starts with _ private
            # and don't document it
            output.append( class_header.format(cl[0])) 
            # Get the docstring
            output.append(pydoc.inspect.getdoc(cl[1]))
            # Get the functions
            output.extend(getfunctions(cl[1]))
            # Recurse into any subclasses
            output.extend(getclasses(cl[1], exclude))
            output.append('\n')
    
    return output


def getfunctions(item):
    output = list()
    #print item
    for func in pydoc.inspect.getmembers(item, pydoc.inspect.isfunction):
        
        if func[0].startswith('_') and func[0] != '__init__':
            continue

        output.append(function_header.format(func[0].replace('_', '\\_')))

        # Get the signature
        output.append ('```py\n')
        output.append('def %s%s\n' % (func[0], pydoc.inspect.formatargspec(*pydoc.inspect.getargspec(func[1]))))
        output.append ('```\n')

        # get the docstring
        if pydoc.inspect.getdoc(func[1]):
            output.append('\n')
            output.append(pydoc.inspect.getdoc(func[1]))

        output.append('\n')
    return output

def generatedocs(module, exclude=[]):
    try:
        sys.path.append(os.getcwd())
        # Attempt import
        mod = pydoc.safeimport(module)
        if mod is None:
           print("Module not found")
        
        # Module imported correctly, let's create the docs
        return getmarkdown(mod, exclude)
    except pydoc.ErrorDuringImport as e:
        print("Error while trying to import " + str(module))

#if __name__ == '__main__':
#
#    print(generatedocs(sys.argv[1]))
#view rawgendocs.py hosted with by GitHub
fid = open('../doc/docs/abm4py.md','w')
fid.write(generatedocs("abm4py"))
fid.close()

fid = open('../doc/docs/agent.md','w')
fid.write(generatedocs("abm4py.agent", exclude=['Parallel'] ))
fid.close()

fid = open('../doc/docs/core.md','w')
fid.write(generatedocs("abm4py.core"))
fid.close()

fid = open('../doc/docs/traits.md','w')
fid.write(generatedocs("abm4py.traits"))
fid.close()

fid = open('../doc/docs/future_traits.md','w')
fid.write(generatedocs("abm4py.future_traits"))
fid.close()

fid = open('../doc/docs/graph.md','w')
fid.write(generatedocs("abm4py.graph", exclude=[ ]))
fid.close()

fid = open('../doc/docs/location.md','w')
fid.write(generatedocs("abm4py.location", exclude=['Parallel']))
fid.close()

fid = open('../doc/docs/misc.md','w')
fid.write(generatedocs("abm4py.misc"))
fid.close()

fid = open('../doc/docs/world.md','w')
fid.write(generatedocs("abm4py.world", exclude=["ABMGraph", ]))
fid.close()