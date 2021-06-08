import xml.etree.ElementTree as ET
from torch_geometric.datasets import ZINC
import os

MUTA_dict = {'S': 0, 'Li': 1, 'K': 2, 'H': 3, 'N': 4, 'C': 5, 'Na': 6, 'Ca': 7, 'Br': 8, 'O': 9, 'P': 10, 'F': 11, 'I': 12, 'Cl': 13}



def get_label(addres):
    return {x.get('file'): (x.get('class')) for x in ET.parse(addres).getroot().iter("print")}

ENZYMES_labels = AIDS_labels = MUTA_labels = None


def read_ENZYMES(name, node_type=True, edge_type=True):
    global ENZYMES_labels
    if not ENZYMES_labels:
        ENZYMES_labels = {k: int(v)-1 for k, v in get_label("./Protein-GED/Protein/labels").items()}
    tree_gxl = ET.parse("./Protein-GED/Protein/" + name)
    root_gxl = tree_gxl.getroot()
    node_label = {}

    # Parse nodes
    for i, node in enumerate(root_gxl.iter('node')):
        attrs = []
        if node_type:
            for attr in node.iter('attr'):
                if (attr.get('name') == "type"):
                    # if (attr.get('name') == "type" or attr.get('name') == "aaLength"):
                    attrs.append(int(attr.find('int').text))
                # if (attr.get('name') == 'x'):
                #     x = float(attr.find('float').text)
                # elif (attr.get('name') == 'y'):
                #     y = float(attr.find('float').text)
        node_label[int(node.get('id'))] = attrs[0] if len(attrs) != 0 else 0

    edges = set()
    edge_label = {}

    for edge in root_gxl.iter('edge'):
        s = int(edge.get('from'))
        t = int(edge.get('to'))

        # Undirected Graph
        edges.add((s, t))
        attrs = []
        if edge_type:
            for attr in edge.iter('attr'):
                if (attr.get('name') == 'frequency'):
                    attrs.append(int(attr.find('int').text))
                # else:
                #     attrs.append(float(attr.find('double').text))
            edge_label[(s, t)] = attrs[0]
    # Create the networkx graph

    return [name, [edges, node_label, edge_label], ENZYMES_labels[name]]


def read_AIDS(name):
    global AIDS_labels
    if not AIDS_labels:
        AIDS_labels = {k: 0 if v == "a" else 1 for k, v in get_label("./AIDS/data/labels").items()}
    tree_gxl = ET.parse("./AIDS/data/" + name)
    root_gxl = tree_gxl.getroot()
    node_label = {}

    # Parse nodes
    for i, node in enumerate(root_gxl.iter('node')):
        attrs = []
        for attr in node.iter('attr'):
            if (attr.get('name') == "chem"):
                # if (attr.get('name') == "type" or attr.get('name') == "aaLength"):
                attrs.append(int(attr.find('int').text))
            # if (attr.get('name') == 'x'):
            #     x = float(attr.find('float').text)
            # elif (attr.get('name') == 'y'):
            #     y = float(attr.find('float').text)
        node_label[int(node.get('id').strip("_"))] = attrs[0] if len(attrs) != 0 else 0

    edges = set()
    edge_label = {}

    for edge in root_gxl.iter('edge'):
        s = int(edge.get('from').strip("_"))
        t = int(edge.get('to').strip("_"))

        # Undirected Graph
        edges.add((s, t))
        attrs = []
        for attr in edge.iter('attr'):
            if (attr.get('name') == 'valence'):
                attrs.append(int(attr.find('int').text))
            # else:
            #     attrs.append(float(attr.find('double').text))
        edge_label[(s, t)] = attrs[0]
    # Create the networkx graph

    return [name, [edges, node_label, edge_label], AIDS_labels[name]]


def read_MUTA(name):
    global MUTA_labels
    if not MUTA_labels:
        MUTA_labels = {k: 0 if v == "mutagen" else 1 for k, v in get_label("./MUTA/data/labels").items()}
    tree_gxl = ET.parse("./MUTA/data/" + name)
    root_gxl = tree_gxl.getroot()
    node_label = {}

    # Parse nodes
    for i, node in enumerate(root_gxl.iter('node')):
        attrs = []
        for attr in node.iter('attr'):
            if (attr.get('name') == "chem"):
                # if (attr.get('name') == "type" or attr.get('name') == "aaLength"):
                attrs.append(MUTA_dict[attr.find('string').text])
            # if (attr.get('name') == 'x'):
            #     x = float(attr.find('float').text)
            # elif (attr.get('name') == 'y'):
            #     y = float(attr.find('float').text)
        node_label[int(node.get('id').strip("_"))] = attrs[0] if len(attrs) != 0 else 0

    edges = set()
    edge_label = {}

    for edge in root_gxl.iter('edge'):
        s = int(edge.get('from').strip("_"))
        t = int(edge.get('to').strip("_"))

        # Undirected Graph
        edges.add((s, t))
        attrs = []
        for attr in edge.iter('attr'):
            if (attr.get('name') == 'valence'):
                attrs.append(int(attr.find('int').text))
            # else:
            #     attrs.append(float(attr.find('double').text))
        edge_label[(s, t)] = attrs[0]
    # Create the networkx graph

    return [name, [edges, node_label, edge_label], MUTA_labels[name]]


def read_ZINC():
    dataset = ZINC(root='./Zinc_dataset',subset=True)
    temp = []
    for i,x in enumerate(dataset):
        edge_attrs ={(int(x),int(y)):int(z) for x,y,z in zip(x.edge_index[0],x.edge_index[1],x.edge_attr)}
        edges = set(edge_attrs.keys())
        nodes = {i:int(x) for i,x in enumerate(x.x)}
        temp.append([i,[edges,nodes,edge_attrs]])
    return temp

def read_ENZYMES_sp(dataset):
    temp = []
    for i,x in enumerate(dataset):
        edge_attrs ={(int(x),int(y)):0 for x,y in zip(x.edge_index[0],x.edge_index[1])}
        edges = set(edge_attrs.keys())
        nodes = {i:int("0b"+"".join([str(int(y)) for y in x]),2) for i,x in enumerate(x.x)}
        temp.append([i,[edges,nodes,edge_attrs]])
    return temp

def getData(name):
    if name == "ENZYMES":
        return [read_ENZYMES("enzyme_" + str(x) + ".gxl") for x in range(1, 601)]
    elif name == "AIDS":
        return [read_AIDS(x) for x in os.listdir("./AIDS/data") if "gxl" in x]
    elif name == "MUTA":
        return [read_MUTA("molecule_"+str(x)+".gxl") for x in range(1,4338)]
    elif name == "ZINC":
        return read_ZINC()