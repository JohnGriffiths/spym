def fix_docs(docs, fix):
    fixed_docs = []
    
    for doc in docs:
        fixed_doc = {}

        for key in doc.keys():
            if key not in ['t_maps', 'c_maps', 'b_maps']:
                fixed_doc[key] = doc[key]

        for name in doc['t_maps'].keys():
            if name in fix.keys():
                fixed_doc.setdefault(
                    't_maps', 
                    {}).setdefault(fix[name], doc['t_maps'][name])

        for name in doc['c_maps'].keys():
            if name in fix.keys():
                fixed_doc.setdefault(
                    'c_maps', 
                    {}).setdefault(fix[name], doc['c_maps'][name])

        fixed_docs.append(fixed_doc)

    return fixed_docs
