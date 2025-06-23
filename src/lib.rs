// src/lib.rs

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
// ADDED: serde for serialization
use serde::{Serialize, Deserialize};


pub mod text_processing;
pub mod generative_model;


// --- Foundational Structs ---

// ADDED: Derive Serialize and Deserialize
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Distinction {
    id: u64,
    self_reference: u64,
}

// ADDED: Derive Serialize and Deserialize
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceStructure {
    id: u64,
    information: u64,
    source: Distinction,
    target: Distinction,
    depth: u32,
}

// ADDED: Derive Serialize and Deserialize
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pattern {
    id: u64,
    stability: u64,
    structure_ids: Vec<u64>,
}

// ADDED: Derive Serialize and Deserialize
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfModel {
    id: u64,
    pub quality: u64,
    pub complexity: u64,
    known_pattern_ids: Vec<u64>,
}

// ADDED: Derive Serialize and Deserialize
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    id: u64,
    center: Distinction,
    integration_hash: u64,
    pub structures: Vec<ReferenceStructure>,
    patterns: Vec<Pattern>,
    history: Vec<u64>,
    model: SelfModel,
}

// --- Implementations ---

impl Distinction {
    pub fn new(seed: &str) -> Self {
        let id = Self::calculate_hash(seed);
        let self_reference = Self::calculate_hash(&id);
        Distinction { id, self_reference }
    }

    pub fn multiply(&self) -> Distinction {
        let new_seed = self.id.to_string();
        Distinction::new(&new_seed)
    }

    pub fn form_frame(&self) -> Frame {
        Frame::new(self)
    }

    pub fn calculate_hash<T: Hash + ?Sized>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    pub fn id(&self) -> u64 { self.id }
}

impl ReferenceStructure {
    /// Internal constructor used for standard evolution.
    fn from_source(source: &Distinction) -> Self {
        let target = source.multiply();
        let depth = 1;
        let id = Self::calculate_id(source, &target, depth);
        let information = Self::calculate_information(id);
        ReferenceStructure { id, information, source: source.clone(), target, depth }
    }

    /// A new public constructor for building a structure from two known points.
    /// This resolves the duplicate definition error.
    pub fn from_source_and_target(source: &Distinction, target: &Distinction, depth: u32) -> Self {
        let id = Self::calculate_id(source, target, depth);
        let information = Self::calculate_information(id);
        ReferenceStructure { id, information, source: source.clone(), target: target.clone(), depth }
    }

    pub fn extend_with_seed(&self, seed: u64) -> ReferenceStructure {
        let new_source = &self.target;
        let new_target = Distinction::new(&seed.to_string());
        let new_depth = self.depth + 1;
        let id = Self::calculate_id(new_source, &new_target, new_depth);
        let information = Self::calculate_information(id);
        ReferenceStructure { id, information, source: new_source.clone(), target: new_target, depth: new_depth }
    }

    fn calculate_id(source: &Distinction, target: &Distinction, depth: u32) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher); target.hash(&mut hasher); depth.hash(&mut hasher);
        hasher.finish()
    }

    fn calculate_information(id: u64) -> u64 { id % 7 }
    pub fn id(&self) -> u64 { self.id }
    pub fn source(&self) -> &Distinction { &self.source }
    pub fn target(&self) -> &Distinction { &self.target }
}

impl Frame {
    /// The primary constructor for a new, "blank" frame.
    pub fn new(root: &Distinction) -> Self {
        let mut frame = Frame {
            id: 0,
            center: Distinction::new("center_placeholder"),
            integration_hash: 0,
            structures: vec![ReferenceStructure::from_source(root)],
            patterns: Vec::new(),
            history: Vec::new(),
            model: SelfModel { id: 0, quality: 0, complexity: 0, known_pattern_ids: Vec::new() },
        };
        frame.update_identity();
        frame
    }

    /// A new public constructor to build a Frame from a pre-computed list of structures.
    pub fn from_structures(root: Distinction, structures: Vec<ReferenceStructure>) -> Self {
        let mut frame = Frame {
            id: 0,
            center: root,
            integration_hash: 0,
            structures,
            patterns: Vec::new(),
            history: Vec::new(),
            model: SelfModel { id: 0, quality: 0, complexity: 0, known_pattern_ids: Vec::new() },
        };
        frame.update_identity();
        frame
    }

    pub fn evolve(&mut self) {
        if let Some(last_structure) = self.structures.last().cloned() {
            let seed = last_structure.id ^ self.model.quality;
            let new_structure = last_structure.extend_with_seed(seed);
            self.structures.push(new_structure);
            self.update_identity();
        }
    }

    pub fn generate(&self) -> Frame {
        let seed = self.id ^ self.history.last().cloned().unwrap_or(0);
        let root_distinction = Distinction::new(&seed.to_string());
        Frame::new(&root_distinction)
    }

    pub fn update_identity(&mut self) {
        if self.integration_hash != 0 { self.history.push(self.integration_hash); }
        self.identify_patterns();
        self.model = self.build_self_model();
        self.id = Self::calculate_frame_id(self);
        self.center = Distinction::new(&self.id.to_string());
        self.integration_hash = self.calculate_integration_hash();
    }

    fn identify_patterns(&mut self) {
        let mut info_map: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for s in &self.structures { info_map.entry(s.information).or_default().push(s.id()); }
        self.patterns.clear();
        for (_info_value, ids) in info_map {
            if ids.len() > 1 {
                self.patterns.push(Pattern {
                    id: Self::calculate_hash_from_slice(&ids),
                    stability: ids.len() as u64,
                    structure_ids: ids,
                });
            }
        }
    }

    fn build_self_model(&self) -> SelfModel {
        let known_pattern_ids: Vec<u64> = self.patterns.iter().map(|p| p.id).collect();
        let quality: u64 = self.patterns.iter().map(|p| p.stability).sum();
        let complexity: u64 = self.structures.len() as u64;
        let mut hasher = DefaultHasher::new();
        known_pattern_ids.hash(&mut hasher); quality.hash(&mut hasher); complexity.hash(&mut hasher);
        SelfModel { id: hasher.finish(), quality, complexity, known_pattern_ids }
    }

    fn calculate_frame_id(frame: &Frame) -> u64 {
        let mut hasher = DefaultHasher::new();
        frame.structures.hash(&mut hasher); frame.model.hash(&mut hasher);
        hasher.finish()
    }

    fn calculate_integration_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher); self.center.hash(&mut hasher);
        hasher.finish()
    }

    fn calculate_hash_from_slice<T: Hash>(slice: &[T]) -> u64 {
        let mut hasher = DefaultHasher::new();
        slice.hash(&mut hasher);
        hasher.finish()
    }

    pub fn is_integrated(&self) -> bool { self.integration_hash == self.calculate_integration_hash() }
    pub fn id(&self) -> u64 { self.id }
    pub fn model(&self) -> &SelfModel { &self.model }
}

pub fn interact(influencer: &Frame, receiver: &mut Frame) {
    if !influencer.is_integrated() || !receiver.is_integrated() { return; }
    if let Some(last_structure) = receiver.structures.last().cloned() {
        let seed = last_structure.id ^ influencer.model.quality;
        let new_structure = last_structure.extend_with_seed(seed);
        receiver.structures.push(new_structure);
        receiver.update_identity();
    }
}
