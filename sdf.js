function my_levenshtein(s1, s2) {
	if (s1.length == s2.length) {
		if (s1 == s2) {
			return 0;
		}
		let count = 0;
		for (i = 0; i < s1.length; i++) {
			if (s1[i] != s2[i]) {
				count++;
			}
		}
		return count;
	}
	return -1;
}
