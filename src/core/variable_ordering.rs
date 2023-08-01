use std::ops::Index;

use crate::core::key::Key;
use hashbrown::HashMap;
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone)]
pub struct VariableOrdering {
    keymap: HashMap<Key, usize>,
    keylist: Vec<Key>,
}
impl VariableOrdering {
    pub fn key(&self, index: usize) -> Key {
        self.keylist[index]
    }

    pub fn len(&self) -> usize {
        self.keylist.len()
    }

    pub fn keys(&self) -> &[Key] {
        &self.keylist
    }
}
impl Default for VariableOrdering {
    fn default() -> Self {
        VariableOrdering {
            keymap: HashMap::<Key, usize>::new(),
            keylist: Vec::<Key>::new(),
        }
    }
}
impl VariableOrdering {
    pub fn new(keylist: &[Key]) -> Self {
        let mut keymap: HashMap<Key, usize> = HashMap::new();
        for (i, k) in keylist.iter().enumerate() {
            keymap.insert(*k, i);
        }
        VariableOrdering {
            keymap,
            keylist: keylist.to_owned(),
        }
    }
}
impl Index<usize> for VariableOrdering {
    type Output = Key;

    fn index(&self, index: usize) -> &Self::Output {
        &self.keylist[index]
    }
}
