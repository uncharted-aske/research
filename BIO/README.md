# Notes on INDRA Ontology

> there isn’t a full lookup table but the main ones are listed here: https://indra.readthedocs.io/en/latest/modules/statements.html
> the space of db_refs is not strictly constrained, we allow a variant of any namespace in the http://identifiers.org/ registry, which is in the hundreds or maybe even thousands
> one thing that might be of interest is the IndraOntology graph which is a single graph that contains each relevant ontology’s hierarchical structure so you might be able to make use of that without having to implement something for each specific third-party ontology
> the technical documentation is here: https://indra.readthedocs.io/en/latest/modules/ontology/bio_ontology.html
> currently, the IndraOntology is available as a Python object that can be imported from INDRA, but if you’re working in a different environment, I could dump a serialization of the graph that you can load
> The ontology is versioned and I am constantly working on improving it, in fact I have a branch open for making a number of improvements right now. The current version is 1.3 and this is  the ontology used to assemble the latest EMMAA COVID-19 model.
> Each node has an id of the form “NAMESPACE:ID” and wherever necessary, the node also has a name which is its standard name in the given namespace. Note that some namespaces include the namespace in the ID so you might see something like CHEBI:CHEBI:12345.
> The graph has 3 types of edges: xref, isa and partof.
> You generally should not use xref links - these are applied by INDRA during assembly to standardize IDs in db_refs but once this standardization is done, you should generally not try to resolve any xref links downstream of INDRA; take whatever is in db_refs for a given agent instead.
> The isa and partof links are the ones representing the hierachies within each ontology (or to me more precise, they are usually within a given namespace but in some cases involve 2 or more namespaces, e.g., HGNC/UP/FPLX for proteins and families).
> It is also important to note that there is an explicit priority order among namespaces that can appear in an Agent’s db_refs and in any situation where you want to choose a single canonical namespace and ID for an agent, you should use this order to be consistent with how the model was assembled. This order is currently [‘FPLX’, ‘UPPRO’, ‘HGNC’, ‘UP’, ‘CHEBI’, ‘GO’, ‘MESH’, ‘MIRBASE’, ‘DOID’, ‘HP’, ‘EFO’]




