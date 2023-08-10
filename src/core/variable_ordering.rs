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
    pub fn key(&self, index: usize) -> Option<Key> {
        self.keylist.get(index).copied()
    }
    pub fn len(&self) -> usize {
        self.keylist.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keylist.is_empty()
    }
    pub fn keys(&self) -> &[Key] {
        &self.keylist
    }
    pub fn search_key(&self, key: Key) -> Option<usize> {
        self.keymap.get(&key).copied()
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
            if keymap.insert(*k, i).is_some() {
                panic!("keymap already has a key {:?}", k)
            }
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
#[cfg(test)]
mod tests {
    use crate::core::key::Key;

    use super::VariableOrdering;

    #[test]
    fn key() {
        let keys = vec![Key(0), Key(1)];
        let var_ord = VariableOrdering::new(&keys);
        assert_eq!(var_ord.key(0).unwrap(), Key(0));
        assert_eq!(var_ord.key(1).unwrap(), Key(1));
        assert!(var_ord.key(2).is_none());
    }
    #[test]
    fn len() {
        let keys = vec![Key(0), Key(1)];
        let var_ord = VariableOrdering::new(&keys);
        assert_eq!(var_ord.len(), 2);
    }
    #[test]
    fn search_key() {
        let keys = vec![Key(0), Key(1)];
        let var_ord = VariableOrdering::new(&keys);
        assert_eq!(var_ord.search_key(Key(0)).unwrap(), 0);
        assert_eq!(var_ord.search_key(Key(1)).unwrap(), 1);
    }
    #[test]
    #[should_panic]
    fn new_with_duplicate_keys() {
        let keys = vec![Key(0), Key(1), Key(1)];
        let _var_ord = VariableOrdering::new(&keys);
    }
}
