const { exp, sum } = require('../math.min');

exports.relu = (input) => {
	return input.map((row) => {
		return row.map((ele) => (ele >= 0 ? ele : 0));
	});
};

//[sample, 10]
exports.softmax = (input) => {
	return input.map((row) => {
		let x = sum(exp(row));
		return row.map((ele) => exp(ele) / x);
	});
};

exports.reluDerivate = (input) => {
	return input.map((row) => {
		return row.map((ele) => (ele >= 0 ? 1 : 0));
	});
};
