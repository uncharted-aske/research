declare namespace GroMEt {
    
    
    /**
     * Metadatum types:
     * (*) <Any>.CodeSpanReference
     * (*) <Gromet>.ModelDescription  # provides textual description and name of the model
     * (*) <Gromet>.ModelInterface  # designates variables, parameters, initial_conditions
     * (*) <Gromet>.TextualDocumentReferenceSet
     * (*) <Gromet>.CodeCollectionReference
     * (*) <Box>.EquationDefinition
     * (*) <Variable>.TextDescription
     * (*) <Variable>.TextParameter
     * () <Variable>.EquationParameter
     *
     * 
     * INDRA Metadatum types:
     * () <Junction>.ReactionReference
     * () <Junction>.IndraAgentReferenceSet
     */
    
    
    //  =============================================================================
    //  Uid
    //  =============================================================================
    
    //  The following also get defined in gromet.py, which imports this file...
    type UidVariable = string;
    type UidJunction = string;
    
    type UidMetadatum = string;
    type UidDocumentReference = string;
    type UidCodeFileReference = string;
    
    //  ISO 8601 Extended format: YYYY-MM-DDTHH:mm:ss:ffffff_ZZZ±zzzz
    //  where
    //  YYYY : 4-digit year
    //  MM   : 2-digit month (January is 01, December is 12)
    //  DD   : 2-digit date (0 to 31)
    //  -    : Date delimiters
    //  T    : Indicates the start of time
    //  HH   : 24-digit hour (0 to 23)
    //  mm   : Minutes (0 to 59)
    //  ss   : Seconds (0 to 59)
    //  ffffff  : Microseconds (0 to 999)
    //  :    : Time delimiters
    //  _    : Time zone delimiter
    //  ZZZ  : Three-letter timezone
    //  zzzz : 4 number UTC timezone offset
    type Datetime = string;
    
    //  for tz in pytz.common_timezones:
    //      print(tz)
    
    
    //  TODO Description of method that produced the metadata
    //  Some descriptor that enables identifying process by which metadata was created
    //  There will generally be a set of 1 or more methods that may generate
    //  each Metadatum type
    //  For example: AutoMATES program analysis creates code_span metadata.
    type MetadatumMethod = string;
    
    
    //  -----------------------------------------------------------------------------
    
    /**
     *     Utility for getting a GroMEt formatted current datetime string.
     *     String is in Datetime ISO 8601 Extended format format (see comment above)
     *         YYYY-MM-DDTHH:mm:ss:ffffff_ZZZ±zzzz
     *     (helpful resource https://pythonhosted.org/pytz/)
     *     :tz: Specify pytz timezone
     *     :return: Datetime
     */
    
    
    //  =============================================================================
    //  Metadatum
    //  =============================================================================
    
    interface MetadatumElm {
        /**
         * Base class for all Gromet Metadatum types.
         * Implements __post_init__ that saves syntactic type (syntax)
         *     as GroMEt element class name.
         */
        metadata_type: string;
    }

    interface Provenance extends MetadatumElm {
        /**
         * Provenance of metadata
         */
        method: MetadatumMethod;
        timestamp: Datetime;
    }

    interface Metadatum extends MetadatumElm {
        /**
         * Metadatum base.
         */
        uid: UidMetadatum;
        provenance: Provenance;
    }

    //  TODO: add Metadatum subtypes
    //        Will be based on: https://ml4ai.github.io/automates-v2/grfn_metadata.html
    
    
    type Metadata = Metadatum[] | null;
    
    
    //  MetadatumAny = NewType('MetadatumAny', Metadatum)
    interface MetadatumAny extends Metadatum {
    }

    //  MetadatumGromet = NewType('MetadatumGromet', Metadatum)
    interface MetadatumGromet extends Metadatum {
    }

    //  MetadatumBox = NewType('MetadatumBox', Metadatum)
    interface MetadatumBox extends Metadatum {
    }

    //  MetadatumVariable = NewType('MetadatumVariable', Metadatum)
    interface MetadatumVariable extends Metadatum {
    }

    //  MetadatumJunction = NewType('MetadatumJunction', Metadatum)
    interface MetadatumJunction extends Metadatum {
    }

    //  =============================================================================
    //  Metadata components
    //  =============================================================================
    
    interface TextSpan {
        page: number;
        block: number;
        char_begin: number;
        char_end: number;
    }

    interface TextExtraction {
        /**
         * Text extraction.
         * 'document_reference_uid' should match the uid of a
         *   TextualDocumentReference for the document from which this
         *   text definition was extracted.
         * COSMOS within-document reference coordinates to the span of text.
         *   'block' is found on a 'page'
         *   'char_begin' and 'char_end' are relative to the 'block'.
         */
        document_reference_uid: UidDocumentReference;
        text_spans: TextSpan[];
    }

    interface EquationExtraction {
        /**
         * 'document_reference_uid' should match the uid of a
         *   TextualDocumentReference.
         * 'equation_number' is 0-indexed, relative order of equation
         *   as identified in the document.
         */
        document_reference_uid: UidDocumentReference;
        equation_number: number;
        equation_source_latex: string; // latex
        equation_source_mml: string; // MathML
    }

    interface CodeFileReference {
        /**
         * 'name': filename
         * 'path': Assume starting from root of code collection
         */
        uid: UidCodeFileReference;
        name: string;
        path: string;
    }

    //  =============================================================================
    //  Metadata host: <Any>
    //  metadata that may be associated with any GroMEt element
    //  =============================================================================
    
    //  -----------------------------------------------------------------------------
    //  CodeSpanReference
    //  -----------------------------------------------------------------------------
    
    interface CodeSpanReference extends MetadatumAny {
        /**
         * host: <Any>
         * Code span references may be associated with any GroMEt object.
         * 'code_type': One of 'IDENTIFIER', 'CODE_BLOCK'
         * code span coordinates are relative to the source file
         *     (denoted by the file_id)
         */
        code_type: string; // 'IDENTIFIER', 'CODE_BLOCK'
        file_id: UidCodeFileReference;
        line_begin: number;
        line_end: number | null; // None if one one line
        col_begin: number | null; // None if multi-line
        col_end: number | null; // None if single char or multi-line
    }

    //  =============================================================================
    //  Metadata host: <Gromet>
    //  metadata associated with a top-level <Gromet> object
    //  =============================================================================
    
    //  -----------------------------------------------------------------------------
    //  ModelInterface
    //  -----------------------------------------------------------------------------
    
    interface ModelDescription extends MetadatumGromet {
        /**
         * host: <Gromet>
         * Provides summary textual description of the model
         *     along with a human-readable name.
         */
        name: string;
        description: string;
    }

    interface ModelInterface extends MetadatumGromet {
        /**
         * host: <Gromet>
         * Explicit definition of model interface.
         * The interface identifies explicit roles of these variables
         * 'variables': All model variables (anything that can be measured)
         * 'parameters': Variables that are generally set to explicit values
         *     (either by default or in experiment spec).
         *     Often these remain constant during execution/simulation,
         *     although they may be updated by the model during
         *     execution/simulation depending on conditions.
         * 'initial_conditions': Variables that typically take an initial
         *     value but then update during execution/simulation
         * TODO: will want to later introduce experiment spec concept
         *         of intervention clamping (keeping parameters/variables
         *         throughout irrespective of original model variable
         *         value update structure).
         */
        variables: Array<UidVariable | UidJunction>;
        parameters: Array<UidVariable | UidJunction>;
        initial_conditions: Array<UidVariable | UidJunction>;
    }

    //  -----------------------------------------------------------------------------
    //  TextualDocumentReferenceSet
    //  -----------------------------------------------------------------------------
    
    //  GlobalReferenceId: Identifier of source document.
    //  Rank preference of identifier type:
    //   (1) 'DOI' (digital objectd identifier) recognize by COSMOS
    //   (2) 'PMID' (Pubmed ID) or other DOI
    //   (3) 'aske_id' (ASKE unique identifier)
    interface GlobalReferenceId {
        type: string;
        id: string;
    }

    interface BibjsonAuthor {
        name: string;
    }

    interface BibjsonIdentifier {
        type: string;
        id: string;
    }

    interface BibjsonLinkObject {
        type: string; // will become: BibjsonLinkType
        location: string;
    }

    interface Bibjson {
        /**
         * Placeholder for bibjson JSON object; format described in:
         *     http://okfnlabs.org/bibjson/
         */
        title: string | null;
        author: BibjsonAuthor[];
        type: string | null;
        website: BibjsonLinkObject;
        timestamp: string | null;
        link: BibjsonLinkObject[];
        identifier: BibjsonIdentifier[];
    }

    interface TextualDocumentReference {
        /**
         * Reference to an individual document
         * 'cosmos_id': ID of COSMOS component used to process document.
         * 'cosmos_version_number': Version number of COSMOS component.
         * 'automates_id': ID of AutoMATES component used to process document.
         * 'automates_version_number': Version number of AutoMATES component.
         */
        uid: UidDocumentReference;
        global_reference_id: GlobalReferenceId;
        cosmos_id: string | null;
        cosmos_version_number: string | null;
        automates_id: string | null;
        automates_version_number: string | null;
        bibjson: Bibjson;
    }

    interface TextualDocumentReferenceSet extends MetadatumGromet {
        /**
         * host: <Gromet>
         * A collection of references to textual documents
         * (e.g., software documentation, scientific publications, etc.).
         */
        documents: TextualDocumentReference[];
    }

    //  -----------------------------------------------------------------------------
    //  CodeCollectionReference
    //  -----------------------------------------------------------------------------
    
    interface CodeCollectionReference extends MetadatumGromet {
        /**
         * host: <Gromet>
         * Reference to a code collection (i.e., repository)
         */
        global_reference_id: GlobalReferenceId;
        file_ids: CodeFileReference[];
    }

    //  =============================================================================
    //  Metadata host: <Box>
    //  metadata associated with a Box
    //  =============================================================================
    
    //  -----------------------------------------------------------------------------
    //  EquationDefinition
    //  -----------------------------------------------------------------------------
    
    interface EquationDefinition extends MetadatumBox {
        /**
         * host: <Box>
         * Association of an equation extraction with a Box
         *     (e.g., Function, Expression, Relation).
         */
        equation_extraction: EquationExtraction;
    }

    //  =============================================================================
    //  Metadata host: <Variable>
    //  metadata associated with a Variable
    //  =============================================================================
    
    //  -----------------------------------------------------------------------------
    //  TextDescription
    //  -----------------------------------------------------------------------------
    
    interface TextDescription extends MetadatumVariable {
        /**
         * host: <Variable>
         * Association of text description of host derived from text source.
         * 'variable_identifier': char/string representation of the variable.
         * 'variable_definition': text definition of the variable.
         * 'description_type': type of description, e.g., definition
         */
        text_extraction: TextExtraction;
        variable_identifier: string;
        variable_description: string;
        description_type: string;
    }

    //  -----------------------------------------------------------------------------
    //  TextParameter
    //  -----------------------------------------------------------------------------
    
    interface TextParameter extends MetadatumVariable {
        /**
         * host: <Variable>
         * Association of parameter values extracted from text.
         */
        text_extraction: TextExtraction;
        variable_identifier: string;
        value: string; // eventually Literal?
    }

    //  -----------------------------------------------------------------------------
    //  TextParameter
    //  -----------------------------------------------------------------------------
    
    interface TextUnit extends MetadatumVariable {
        /**
         * host: <Variable>
         * Association of variable unit type extracted from text.
         */
        text_extraction: TextExtraction;
        unit: string;
    }

    //  -----------------------------------------------------------------------------
    //  EquationParameter
    //  -----------------------------------------------------------------------------
    
    interface EquationParameter extends MetadatumVariable {
        /**
         * host: <Variable>
         * Association of parameter value extracted from equation.
         */
        equation_extraction: EquationExtraction;
        value: string; // eventually Literal?
    }

    //  =============================================================================
    //  Metadata host: <Junction>
    //  metadata associated with a Junction
    //  =============================================================================
    
    //  -----------------------------------------------------------------------------
    //  INDRA Metadatums
    //  -----------------------------------------------------------------------------
    
    interface ReactionReference extends MetadatumJunction {
        /**
         * host: <Junction> : PNC Rate
         */
        indra_stmt_hash: string;
        reaction_rule: string;
        is_reverse: boolean;
    }

    type IndraAgent = { [key: string]: any };
    
    
    interface IndraAgentReferenceSet extends MetadatumJunction {
        /**
         * host: <Junction> : PNC State
         */
        indra_agent_references: IndraAgent[];
    }

    //  =============================================================================
    //  =============================================================================
    //  CHANGE LOG
    //  =============================================================================
    //  =============================================================================
    
    /**
     * Changes 2021-07-10:
     * () Added Variable/Type/Gromet/Variable/Junction/etc. Metadatum sub-types that
     *     we can use to determine which Metdatums that inherit from those are
     *     available at different locations in the GroMEt.
     * () Made the following fields in TextualDocumentReference optional
     *     cosmos_id, cosmos_version_number, automates_id, automates_version_number
     * () Created a BibJSON identifier type: {“type”: str , “id”: str }
     * () Made all fields (except identifier) under Bibjson optional
     * () Changed TextExtraction:
     *     list of: “text_spans”: array, required
     * () Added TextSpan: page, block, char_begin, char_end -- all required
     * () ALL arrays (lists) are required (even if empty)
     *     ONE EXCEPTION: all metadata fields can be None (not updating type)
     * () Updated Bibjson: Removed the “file” and “file_url” in favor of “link”
     *     link is a List of LinkObject types
     * () Added LinkObject: Has the following required fields:
     *     type : currently str, but eventually will be BibJSONLinkType (TODO Paul)
     *         string options include: “url”, “gdrive”, etc
     *     location : a string that can be used to locate whatever is referenced by the Bibjson
     * () Changed TextDescription
     *     Renamed TextDefinition → TextDescription  (including fields)
     *     added field ‘description_type’
     * () Changed EquationParameter
     *     dropped “variable_id” -- not needed (as it is associated in the metadata of the variable
     *
     * 
     *
     * 
     * Changes 2021-06-10:
     * () Started migration of GrFN metadata types to GroMEt metadatum types.
     */
    
    
    /**
     * Shared working examples:
     * gromet/
     *     docs/
     *         <date>-gromet-uml.{png,svg}
     *         <date>-TypedGrometElm-hierarchy-by-hand.pdf
     *         Conditional.pdf  # schematic of structural pattern for Conditional
     *         Loop.pdf         # schematic of structural pattern for Loop
     *     examples/
     *         <dated previous versions of examples>
     *         cond_ex1/        # example Function Network w/ Conditional
     *         Simple_SIR/
     *             Wiring diagrams and JSON for the following Model Framework types
     *               Function Network (FN)
     *               Bilayer
     *               Petri Net Classic (PetriNetClassic)
     *               Predicate/Transition (Pr/T) Petri Net (PrTNet)
     *         toy1/            # example Function Network (no Conditionals or Loops)
     *
     * 
     * (UA:
     * Google Drive:ASKE-AutoMATES/ASKE-E/GroMEt-model-representation-WG/gromet/
     *     root of shared examples
     * Google Drive:ASKE-AutoMATES/ASKE-E/GroMEt-model-representation-WG/gromet-structure-visual
     *     TypedGrometElm-hierarchy-02.graffle
     * )
     *
     * 
     *
     * 
     * TODO: Side Effects
     * () mutations of globals
     *     (can happen in libraries)
     * () mutations of mutable variables
     * () mutations of referenced variables (C/C++, can't occur in Python)
     *
     * 
     * Event-driven programming
     * () No static trace (directed edges from one fn to another),
     *     so a generalization of side-effects
     *     Requires undirected, which corresponds to under-specification
     */
    
    //  -----------------------------------------------------------------------------
    //  Model Framework Types
    //  -----------------------------------------------------------------------------
    
    //  Data:
    //  Real, Float, Integer, Boolean
    
    //  Primitive term constructors (i.e., primitive operators):
    //  arithmetic: "+", "*", "-", "/", "exp", "log"
    //  boolean fn: "lt", "leq", "gr", "geq", "==", "!=", "and", "or", "not"
    
    //  Function Network (FunctionNetwork):
    //  Function, Expression, Predicate, Conditional, Loop,
    //  Junction, Port, Literal, Variable
    //  Types:
    //    Ports: PortInput, PortOutput
    
    //  Bilayer
    //  Junction, Wire
    //  Types:
    //   Junctions: State, Flux, Tangent
    //   Wires: W_in, W_pos, W_neg
    
    //  Petri Net Classic (PetriNetClassic)
    //  Junction, Wire, Literal
    //  Types:
    //   Junction: State, Rate
    
    //  Predicate/Transition (Pr/T) Petri Net (PrTNet)
    //  Relation, Expression, Port, Literal
    //  Types:
    //    Ports: Variable, Parameter
    //    Wire: Undirected  (all wires are undirected, so not strictly required)
    //    Relation: PrTNet, Event, Enable, Rate, Effect
    
    //  -----------------------------------------------------------------------------
    //  GroMEt syntactic types
    //  -----------------------------------------------------------------------------
    
    //  The following gromet spec as a "grammar" is not guaranteed to
    //    be unambiguous.
    //  For this reason, adding explicit "gromet_element" field that
    //    represents the Type of GroMEt syntactic element
    
    
    interface GrometElm {
        /**
         * Base class for all Gromet Elements.
         * Implements __post_init__ that saves syntactic type (syntax)
         *     as GroMEt element class name.
         */
        syntax: string;
    }

    //  --------------------
    //  Uid
    
    //  The purpose here is to provide a kind of "namespace" for the unique IDs
    //  that used to distinguish gromet model component instances.
    //  Currently making these str so I can give them arbitrary names as I
    //    hand-construct example GroMEt instances, but these could be
    //    sequential integers (as James uses) or uuids.
    
    type UidType = string;
    type UidLiteral = string;
    type UidPort = string;
    type UidWire = string;
    type UidBox = string; // Uids for defined Boxes
    type UidOp = string; // Primitive operator name
    type UidGromet = string;
    
    type UidMeasure = string;
    
    
    //  Explicit "reference" objects.
    //  Required when there is ambiguity about which type of uid reference
    //  is specified.
    
    interface RefBox extends GrometElm {
        /**
         * Representation of an explicit reference to a defined box
         */
        name: UidBox;
    }

    interface RefOp extends GrometElm {
        /**
         * Representation of an explicit reference to a primitive operator
         */
        name: UidOp;
    }

    interface RefLiteral extends GrometElm {
        /**
         * Representation of an explicit reference to a declared Literal
         */
        name: UidLiteral;
    }

    //  -----------------------------------------------------------------------------
    //  Type
    //  -----------------------------------------------------------------------------
    
    interface Type {
        /**
         * Type Specification.
         * Constructed as an expression of the GroMEt Type Algebra
         */
        type: string;
    }

    interface TypeDeclaration extends GrometElm {
        name: UidType;
        type: Type;
        metadata: Metadata;
    }

    //  TODO: GroMEt type algebra: "sublangauge" for specifying types
    
    
    //  Atomics
    
    //  Assumed "built-in" Atomic Types:
    //    Any, Void (Nothing)
    //    Number
    //      Integer
    //      Real
    //        Float
    //    Bool
    //    Character
    //    Symbol
    
    //  @dataclass
    //  class Atomic(Type):
    //      pass
    
    
    //  Composites
    
    //  @dataclass
    //  class Composite(Type):
    //      pass
    
    
    //  Algebra
    
    interface Prod extends Type {
        /**
         * A Product type constructor.
         * The elements of the element_type list are assumed to be
         * present in each instance.
         */
        cardinality: number | null;
        element_type: UidType[];
    }

    interface String extends Prod {
        /**
         * A type representing a sequence (Product) of Characters.
         */
        element_type: UidType[];
    }

    interface Sum extends Type {
        /**
         * A Sum type constructor.
         * The elements of the element_type list are assumed to be variants
         * forming a disjoint union; only one variant is actualized in each
         * instance.
         */
        element_type: UidType[];
    }

    interface NamedAttribute extends Type {
        /**
         * A named attribute of a Product composite type.
         */
        name: string;
        element_type: UidType;
    }

    //  @dataclass
    //  class Map(Prod):
    //      element_type: List[Tuple[UidType, UidType]]
    
    
    //  -----------------------------------------------------------------------------
    //  TypedGrometElm
    //  -----------------------------------------------------------------------------
    
    interface TypedGrometElm extends GrometElm {
        /**
         * Base class for all Gromet Elements that may be typed.
         */
        type: UidType | null;
        name: string | null;
        metadata: Metadata;
    }

    //  --------------------
    //  Literal
    
    
    interface Literal extends TypedGrometElm {
        /**
         * Literal base. (A kind of GAT Nullary Term Constructor)
         * A literal is an instance of a Type.
         *
         * 
         * If a Literal is to be "referred to" in multiple places in the model,
         *     then assign it a 'uid' and place its declaration in the
         *     'literals' field of the top-level Gromet.
         *     This is referred to as a 'named Literal'
         *         (where in this case, by "name" I mean the uid)
         * Example of a simple "inline" Literal
         */
        uid: UidLiteral | null; // allows anonymous literals
        value: Val; // TODO
    }

    //  TODO: "sublanguage" for specifying instances
    
    interface Val extends GrometElm {
        val: string | Array<Val | AttributeVal>;
    }

    interface AttributeVal extends GrometElm {
        name: string;
        val: Val;
    }

    /**
     * Interval Number, Number, Number
     *
     * 
     * Type: Pair = Prod(element_type[Int, String]) --> (<int>, <string>)
     * Literal: (type: "Pair", [3, "hello"])
     *
     * 
     * Literal: (type: "Interval", [3, 6.7, 0.001])
     *
     * 
     * SetIntegers = Prod(element_type=[Int])
     * SetIntegers10 = Prod(element_type=[Int], 10)
     * Literal: (type: "SetInt10", [1,2,3,3,4,52....])
     */
    
    
    //  --------------------
    //  Valued
    
    interface Valued extends TypedGrometElm {
        /**
         * This class is never instantiated; it's purpose is to
         *     introduce attributes and a class-grouping into
         *     the class hierarchy.
         * Typed Gromet Elements that may have a 'value'
         * and the 'value_type' determines what types of values
         * the element can have/carry.
         */
        value: Literal | null;
        value_type: UidType | null;
    }

    //  --------------------
    //  Junction
    
    interface Junction extends Valued {
        /**
         * Junction base.
         * Junctions are "0-ary"
         */
        uid: UidJunction;
    }

    //  --------------------
    //  Port
    
    interface Port extends Valued {
        /**
         * Port base.
         * Ports are "1-ary" as they always *must* belong to a single Box
         *     -- you cannot have a Port without a host Box.
         * Ports define an interface to a Box, whereby values may pass from
         *     outside of the Box into the internals of the Box.
         * A Port may be optionally named (e.g., named argument)
         */
        uid: UidPort;
        box: UidBox;
    }

    interface PortCall extends Port {
        /**
         * "Outer" Port of an instance call to a Box definition.
         * There will be a PortCall Port for every Port associated
         *     with the Box definition.
         */
        call: UidPort;
    }

    //  --------------------
    //  Wire
    
    interface Wire extends Valued {
        /**
         * Wire base.
         * Wires are "2-ary" as they connect up to two Valued elements,
         *     the 'src' and the 'tgt'.
         *     Despite the names, 'src' and 'tgt' are NOT inherently
         *         directed.
         *     Whether a Wire is directed depends on its 'type'
         *         within a Model Framework interpretation.
         * All Wires have a 'value_type' (of the value they may carry).
         * Optionally declared with a 'value', otherwise derived
         *     (from system dynamics).
         */
        uid: UidWire;
        src: UidPort | UidJunction | null;
        tgt: UidPort | UidJunction | null;
    }

    //  --------------------
    //  Box
    
    interface Box extends TypedGrometElm {
        /**
         * Box base.
         * A Box may have a name.
         * A Box may have wiring (set of wiring connecting Ports of Boxes)
         */
        uid: UidBox;
        ports: UidPort[] | null;
    }

    interface BoxCall extends Box {
        /**
         * An instance "call" of a Box (the Box definition)
         */
        call: UidBox;
    }

    interface HasContents {
        /**
         * Mixin class, never instantiated.
         * Bookkeeping for Box "contents" references.
         *     Natural to think of boxes "containing" (immediately
         *         contained) Boxes, Junctions and Wires that wire up
         *         the elements.
         *     This information functions like an index and
         *         supports easier identification of the elements
         *         that are the "top level contents" of a Box.
         *     Other Boxes do also have contents, but have special
         *         intended structure that is explicitly represented
         */
        wires: UidWire[] | null;
        boxes: UidBox[] | null;
        junctions: UidJunction[] | null;
    }

    //  Relations
    
    interface Relation extends Box, HasContents { // BoxUndirected
        /**
         * Base Relation
         */
    }

    //  Functions
    
    interface Function extends Box, HasContents { // BoxDirected
        /**
         * Base Function
         * Representations of general functions with contents wiring
         *     inputs to outputs.
         */
    }

    interface Expr extends GrometElm {
        /**
         * Assumption that may need revisiting:
         *   Expr's are assumed to always be declared inline as single
         *     instances, and may include Expr's in their args.
         *   Under this assumption, they do not require a uid or name
         *     -- they are always anonymous single instances.
         * The call field of an Expr is a reference, either to
         *     (a) RefOp: primitive operator.
         *     (b) RefOp: an explicitly defined Box (e.g., a Function)
         * The args field is a list of: UidPort reference, Literal or Expr
         */
        call: RefBox | RefOp;
        args: Array<UidPort | Literal | Expr> | null;
    }

    interface Expression extends Box { // BoxDirected
        /**
         * A BoxDirected who's contents are an expression tree of Exp's.
         * Assumptions:
         *   (1) Any "value" references in the tree will refer to the
         *     input Ports of the Expression. For this reason, there is
         *     no need for Wires.
         *   (2) An Expression always has only one output Port, but for
         *     parity with BoxDirected, the "output_ports" field name
         *     remains plural and is a List (of always one Port).
         */
        tree: Expr;
    }

    interface Predicate extends Expression {
        /**
         * A Predicate is an Expression that has
         *     an assumed Boolean output Port
         *   (although we will not override the parent
         *    BoxDirected parent).
         */
    }

    interface Conditional extends Box {
        /**
         * Conditional
         *     ( TODO:
         *         Assumes no side effects.
         *         Assumes no breaks/continues.
         *     )
         * ( NOTE: the following notes make references to elements as they
         *         appear in Clay's gromet visual notation. )
         * Terminology:
         *     *branch Predicate* (a Predicate is a type of Expression
         *         with a single Boolean PortOutput): represents the branch
         *         conditional test whose outcome determines whether the
         *         branch will be evaluated.
         *     *branch body*: represents the computation of anything in the
         *         branch.
         *         If the branch body only involves computing a single variable,
         *             then it is an Expression.
         *         If the branch body computes more than one variable, then it
         *             is a Function.
         *     A *branch* itself consists of a Tuple of:
         *         (1) branch predicate (Predicate)
         *         (2) branch body (Union[Expression, Function])
         * Port conventions:
         *     A Conditional has a set of PortInput and PortOutput type
         *         Ports/PortCalls that define the Conditional's "input"
         *         and "output" *interface*.
         *     The ports on the Predicates, Expressions and Functions
         *         of the branch predicate and body will (mostly) be
         *         PortCalls that reference the corresponding Ports
         *         in the Conditional Port interface.
         *         (The one exception is the Predicate PortOutput, which
         *         is just a regular Port of value_type Boolean that
         *         itself is determined by the Predicate but does not get
         *         "read" by another model element; it is instead
         *         used in evaluation to determine branch choice.)
         *     *input* Ports (type PortInput) of the Conditional interface
         *         capture any values of variables from the scope
         *         outside of the Conditional Box that are required by any
         *         branch Predicate or body.
         *         (Think of the PortInput Ports as representing the
         *         relevant "variable environment" to the Conditional.)
         *     We can then think of each branch body as a possible
         *         modification to the "variable environment" of the
         *         input Ports. When a branch body Expression/Function is
         *         evaluated, it may: (a) preserve the values from some
         *         or all of the original input ports, or (b) modify the
         *         variable values introduced by those input ports,
         *         and/or (c) introduce *new* variables resulting in
         *         corresponding new output Ports.
         *     From the perspective of the PortOutput type ports of the
         *         Conditional output interface, we need to consider all
         *         of the possible new variable environment changes made
         *         by the selection of any branch.
         *         Representing all possible branch environment variabels
         *         permits us to treat the Conditional as a modular
         *         building-block to other model structures.
         *         The Conditional PortOutput interface will therefore have a
         *         corresponding PortOutput port for each possible variable
         *         introduced in any branch body.
         *     In cases where a PortOutput port of the Conditional represents
         *         a variable that may not be set by a branch (or none of the
         *         branches conditions evaluate to True and there is no *else*
         *         branch body), but the variable represented by that port
         *         *is* represented by a Port in the PortInput interface,
         *         then the PortOutput will be a PortCall that 'call's the
         *         PortInput Port.
         * Evaluation semantics:
         *     Branches are visited in order until the current branch
         *         Predicate evals to True.
         *     If a branch Predicate evaluates to True, then the branch
         *         body Expression/Function uses the values of the
         *         Conditional PortInput Ports referred to ('call'ed by) the
         *         body PortCalls and computes the resulting values of
         *         variables (represented by the PortOutput PortCalls of the
         *         body Expression or Function) in that branch;
         *         The PortOutput PortCalls of the branch then 'call'
         *         the corresponding PortOutput Ports in the Conditional
         *         output interface, setting their values.
         *     Any PortOutput ports NOT called by a branch body will then
         *         either:
         *         (1) themselves be PortCalls that call the corresponding
         *             PortInput interface Ports to retrieve the variable's
         *             original value, or
         *         (2) have undefined (None) values.
         *     An "else" branch has no branch Predicate -- the branch
         *         body is directly evaluated. Only the last branch
         *         may have no branch Predicate.
         *     Finally, if none of the branch Predicates evaluate to True
         *         and there is no "else" branch, then evaluation 'passes':
         *         any PortOutput PortCalls in the Conditional output interface
         *         will get their values from their 'call'ed PortInput,
         *         or have undefined values.
         *
         * 
         *     TODO Updating that branch body MUST have Expression/Function
         */
        //  branches is a List of UidBox references to
        //    ( <Predicate>1, <Expression,Function> )
        branches: Array<[UidBox | null, UidBox]>;
    }

    interface Loop extends Box, HasContents {
        /**
         * Loop
         *     ( TODO:
         *         Assumes no side-effects.
         *         Assumes no breaks.
         *     )
         * A Box that "loops" until an exit_condition (Predicate)
         *     is True.
         *     By "loop", you can think of iteratively making copies of the
         *         Loop and wiring the previous Loop instance PortOutputs
         *         to the PortInput Ports of the next Loop instance.
         *     "Wiring"/correspondence of output-to-input Ports is
         *         accomplished by the PortOutput ports being
         *         PortCalls that directly denote (call) their
         *         corresponding PortInput Port.
         * Terminology:
         *     A Loop has a contents (because it is a HasContents)
         *         -- wires, junctions, boxes -- that represent the
         *         *body* of the loop.
         *     A Loop has an *exit_condition*, a Predicate that
         *         determines whether to evaluate the loop.
         *     A Loop has PortInput and PortOutput ports.
         *         A portion of the PortInput Ports acquire values
         *             by the incoming external "environment"
         *             of the Loop.
         *         The remaining of the PortInput Ports represent
         *             Ports to capture the state variables that may
         *             be introduced within the Loop body but not
         *             originating from the incoming external
         *             "environment".
         *             In the initial evaluation of the loop,
         *             these Ports have no values OR the Port
         *             has an initial 'value' Literal
         *             (e.g., initializing a loop index).
         *             After iteration through the loop, these
         *             Ports may have their values assigned/changed
         *             by the Loop body; these values are then used
         *             to set the PortInput values for the start of
         *             the next iteration through the loop.
         *     Each PortInput Port is "paired" with (called by) a
         *         PortOutput PortCall.
         *     Some PortInput Port values will not be changed as a
         *         result of the Loop body, so these values "pass
         *         through" to that input's paired output PortCall.
         *         Others may be changed by the Loop body evaluation.
         * Evaluation semantics:
         *     The Loop exit_condition Predicate is evaluated at
         *         the very beginning before evaluating any of the
         *         Loop body wiring.
         *         IF True (the exit_condition evaluates to True),
         *             then the PortOutput PortCalls have their
         *             values set by the PortInput Ports they call,
         *             skipping over any intervening computation in
         *             Loop body.
         *         IF False (the exit_condition evaluates to False),
         *             then the Loop body wiring is evaluated to
         *             determine the state of each PortOutput
         *             PortCall value.
         *             The values of each PortOutput PortCall are
         *             then assigned to the PortCall's 'call'ed
         *             PortInput Port and the next Loop iteration
         *             is begun.
         *     This basic semantics supports both standard loop
         *         semantics:
         *         () while: the exit_condition is tested first.
         *         () repeat until: an initial PortInput Port for flagging
         *             the first Loop iteration is set to False to
         *             make the initial exit_condition evaluation fail
         *             and is thereafter set to True in the Loop body.
         */
        exit_condition: UidBox | null;
    }

    //  --------------------
    //  Variable
    
    type VariableState = UidPort | UidWire | UidJunction;
    
    
    interface Variable extends TypedGrometElm {
        /**
         * A Variable is the locus of two representational roles:
         *     (a) denotes one or more elements that are Valued,
         *         i.e., carry a value (aka: states) and
         *     (b) denotes a modeled domain (world) state.
         * Currently, (b) will be represented in Metadata.
         *
         * 
         * 'states' represents the set of model Valued components
         *     that constitute the Variables
         * 'proxy_state' denotes the single model element that can
         *     be used as a representative proxy for the variable
         *     (e.g., visualization).
         *
         * 
         */
        uid: UidVariable;
        proxy_state: VariableState;
        states: VariableState[];
    }

    //  --------------------
    //  Gromet top level class
    
    interface Gromet extends TypedGrometElm {
        uid: UidGromet;
        root: UidBox | null;
        //  definitions
        types: TypeDeclaration[] | null;
        literals: Literal[] | null;
        junctions: Junction[] | null;
        ports: Port[] | null;
        wires: Wire[] | null;
        boxes: Box[] | null;
        variables: Variable[] | null;
    }

    //  -----------------------------------------------------------------------------
    //  Utils
    //  -----------------------------------------------------------------------------
    
    
    
    //  -----------------------------------------------------------------------------
    //  CHANGE LOG
    //  -----------------------------------------------------------------------------
    
    /**
     * Changes 2021-06-22:
     * () Added 'proxy_state' field to store single Valued model component from the
     *     Variable state set that can be used as a proxy for the variable.
     *
     * 
     * Changes 2021-06-21:
     * () Changes to Conditional:
     *     () The conditional branch body may now be either an Expression or Function.
     *     () Explicitly documented that there is no need for any Wires between
     *         the Conditional input Ports and the branch Predicate and body ports,
     *         since these will always have the same corresponding Ports
     *         (similar to how Wires are not needed in Expressions/Expr);
     *         In this case, this is implemented by having the PortInput Ports to the
     *         branch Predicate and the branch body Expression/Function be PortCalls
     *         that 'call' the corresponding Conditional PortInput Ports;
     *         and similarly for branch body output Ports to the corresponding
     *         conditional output Ports: the branch body Expression/Function
     *         output Ports will be PortCalls to the Conditional PortOutput Ports.
     *         And in the case that a branch body does not 'call' one of the defined
     *         Conditional PortOutputs BUT that output port corresponds to an input
     *         port, then that PortOutput will be a PortCall that 'call's the
     *         corresponding PortInput to get its value.
     *     - the 'wiring diagram' schema figure (in the gromet/docs/) has been updated.
     * () Changes to Loop:
     *     () The PortOutput ports will be PortCalls that call their
     *         corresponding PortInput Ports to unambiguously determine
     *         the output-to-input correspondence.
     *         This removes the requirement that the correspondence is
     *         derived from the order in which the Ports are referenced
     *         in the Loop's 'ports' list.
     *
     * 
     * Changes 2021-06-13:
     * () Changed RefFn to RefBox (as reference could be to any defined Box)
     * () Remove UidFn as not needed; instead use general UidBox (e.g., by RefBox)
     * () Moved metadata into separate file to reduce clutter.
     * () Started migration of GrFN metadata types to GroMEt metadatum types.
     *     () <Gromet>.TextualDocumentReferenceSet
     * () First example of Experiment Specification
     *     () ExperimentSpecSet : A set of Experiment Specifications
     *     () FrontendExperimentSpec : Message from HMI to Proxy
     *     () BackendExperimentSpec : Message from Proxy to execution framework
     *
     * 
     *
     * 
     * Changes 2021-05-27:
     * () Added the following mechanism by which a Box can be "called"
     *         in an arbitrary number of different contexts within the gromet.
     *     This include adding the following two TypedGrometElms:
     *     (1) BoxCall: A type of Box --- being a Box, the BoxCall itself has
     *         it's own UidBox uid (to represent the instance) and its own
     *         list of Ports (to wire it within a context).
     *         The BoxCall adds a 'call' field that will consist of the UidBox
     *             of another Box that will serve as the "definition" of the
     *             BoxCall.
     *         An arbitrary number of different BoxCalls may "call" this
     *             "definition" Box. There is nothing else about the
     *             "definition" Box that makes it a definition -- just that
     *             it is being called by a BoxCall.
     *         The BoxCall itself will have no internal contents, it's
     *             internals are defined by the "definition" Box.
     *         For each Port in the "definition" Box, BoxCall will have
     *             a corresponding PortCall Port; this PortCall will reference
     *             the "definition" Box Port.
     *     (2) PortCall: A tye of Port -- being a Port, the PortCall has it's
     *         own UidPort uid (to represent the instance), and adds a 'call'
     *         field that will be the UidPort of the  Port on the "definition"
     *         Box referenced by a BoxCall.
     *         The 'box' field of the PortCall will be the UidBox of the BoxCall
     *             instance.
     *
     * 
     * Changes 2021-05-23:
     * () Added notes at top on Model Framework element Types
     * () Removed Event
     * () Wire tgt -> tgt
     *
     * 
     * Changes 2021-05-21:
     * () Convention change: Model Framework typing will now be represented
     *     exclusively by the 'type' attribute of any TypedGrometElm.
     *     This is a much more clean way of separating syntax (the elements) from
     *         their semantics (how to interpret or differentially visualize).
     *     A Model Framework will designate what types are the TypedGrometElms may be.
     *     For example, a Function Network will have
     *         Port types: PortInput, PortOutput
     *         Wire types: WireDirected, WireUndirected
     *     For example, a Bilayer will have
     *         Port types: PortInput, PortOutput, PortRate
     *         Wire types: W_in, W_pos, W_neg
     *         Junction types; JunctionState, JunctionTangent
     *     In general, a Model Framework type will only be specified when
     *         a distinction between more than one type is needed.
     * () Changes to Wire:
     *     WireDirected and WireUndirected have been removed. All Wires have
     *         input -> src
     *         output -> tgt
     *         Despite the names, these are not necessarily directed, just in-principle
     *             distinction between the two.
     *         The 'type' determines whether a Wire has the property of being directed.
     * () Removed BoxDirected and BoxUndirected.
     *     The "facing" that Ports are associated with will now be represented
     *         in the Port 'type' Model Framework.
     *     Top-level Box now has 'ports' attribute. This is still required
     *         as we need to preserve information about ordering of Ports,
     *         both for positional arguments and for pairing inputs to outputs in Loop.
     * () Valued now includes 'value_type' attribute.
     *     Previously was using Port, Junction and Wire 'type' to capture the
     *         value type, but now the value type will be explicitly represented
     *         by the value_type attribute.
     *     The 'type' attribute will instead be reserved for Model Framework type.
     * () Added 'name' to TypedGrometElm, so all children can be named
     *     The purpose of name: provide model domain-relevant identifier to model component
     * () Metadatum is no longer a TypedGrometElm, just a GrometElm, as it is not
     *     itself a component of a model; it is data *about* a model component.
     * () Gromet object: added 'literals' and 'junctions' attributes
     * TODO:
     *     () Update docs to make explicit the convention for Port positional arguments
     *     () Update Loop docs to make explicit the convention for Port pairing
     *
     * 
     * Changes 2021-05-17:
     * () FOR NOW: commenting out much of the Types, as likely source of
     *     confusion until sorted out.
     * () Added HasContents as a "mixin" that provides explicit lists
     *     of junctions, boxes, and wires.
     *     This is mixed-in to: Loop, Function and Relation.
     * () Added Valued class, a type of TypedGrometElm, that introduced
     *     the attributes of "value" (i.e., for things that can carry
     *     a value). This is now the parent to Junction, Port and Wire.
     *     This also distinguishes those classes from Box, which does
     *         not have a value, but is intended to transform or assert
     *         relationships between values.
     * () Changes to Loop:
     *     Removed "port_map" -- the pairing of input and output Ports
     *         is now based on the order of Ports in the input_ports and
     *         output_ports lists.
     *     Cleaned up documentation to reflect this.
     *
     * 
     * Changes [2021-05-09 to 2021-05-16]:
     * () Added parent class TypedGrometElm inherited by any GrometElm that
     *     has a type. GrometElm's with types play general model structural roles,
     *     while other non-typed GrometElms add element-specific structure (such
     *     as Expr, RefFn, RefOp, Type, etc...)
     * () UidOp will now be reserved ONLY for primitive operators that are not
     *     explicitly defined within the gromet.
     *     All other "implementations" of transformations must have associated
     *         Box definitions with UidBox uids
     * () Introduced RefOp and RefFn to explicitly distinguish between the two
     * () Exp renamed to Expr
     * () Expr field "operator" -> "call", where type is now either RefOp or RefFn
     * () Expression changed from being a child of Function to being child of BoxDirected
     * () Changed Predicate to being an Expression
     * () Added Conditional, child of BoxDirected
     * () Added Loop, child of BoxDirected
     * () WireUndirected
     *     ports changed from List[Union[UidPort, UidJunction]]
     *     to Union[Tuple[UidPort, UidJunction],
     *              Tuple[UidJunction, UidPort],
     *              Tuple[UidPort, UidPort]]
     *     - should only have one pairwise connection per UndirectedWire
     *
     * 
     */
}
